"""
Evaluation layer: runs inference on the test set and computes
a full suite of classification metrics.

After training, call Evaluator.evaluate() to get an EvaluationResult
that can be logged to MLflow and saved as a JSON artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch.utils.data import DataLoader

from pipeline.models.base import ModelWrapper
from pipeline.training.utils import get_device

# Maximum number of misclassified examples to store (keeps artifacts small)
_MAX_MISCLASSIFIED = 20


@dataclass
class EvaluationResult:
    """
    Full test-set evaluation output.

    All fields are plain Python types (no tensors) so the result is
    directly JSON-serialisable via to_dict() / to_json().
    """

    accuracy: float
    precision: float  # macro-averaged
    recall: float  # macro-averaged
    f1: float  # macro-averaged
    per_class_report: dict  # sklearn classification_report(output_dict=True)
    confusion_matrix: list[list[int]]  # row=true, col=predicted
    misclassified_examples: list[dict]  # [{idx, true_label, pred_label}, ...]
    evaluation_time_seconds: float
    num_test_samples: int
    label_names: list[str] | None = field(default=None)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "per_class_report": self.per_class_report,
            "confusion_matrix": self.confusion_matrix,
            "misclassified_examples": self.misclassified_examples,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "num_test_samples": self.num_test_samples,
            "label_names": self.label_names,
        }

    def to_json(self, path: str) -> None:
        """Write the evaluation result to a JSON file at `path`."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary_metrics(self) -> dict[str, float]:
        """Flat dict of the four headline metrics — for MLflow log_metrics()."""
        return {
            "test_accuracy": self.accuracy,
            "test_precision": self.precision,
            "test_recall": self.recall,
            "test_f1": self.f1,
            "evaluation_time_seconds": self.evaluation_time_seconds,
        }


class Evaluator:
    """
    Runs inference on the test dataset and computes classification metrics.

    Args:
        model_wrapper: Trained ModelWrapper (any backend).
        test_dataset:  Tokenized HuggingFace Dataset with torch format.
        batch_size:    Inference batch size (defaults to 32).
        label_names:   Human-readable label names, e.g. ["World", "Sports", ...].
                       If provided, used in the per-class report.
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        test_dataset,
        batch_size: int = 32,
        label_names: list[str] | None = None,
    ) -> None:
        self._wrapper = model_wrapper
        self._test_dataset = test_dataset
        self._batch_size = batch_size
        self._label_names = label_names

    def evaluate(self) -> EvaluationResult:
        """
        Run full test-set evaluation.

        Collects all predictions and ground-truth labels in one pass,
        then computes metrics with sklearn for numerical stability and
        consistency with industry-standard reporting.

        Returns:
            EvaluationResult with accuracy, macro P/R/F1, per-class breakdown,
            confusion matrix, and up to 20 misclassified examples.
        """
        device = get_device()
        model = self._wrapper.get_model()
        model.to(device)
        model.eval()

        loader = DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=False,
        )

        all_preds: list[int] = []
        all_labels: list[int] = []
        all_indices: list[int] = []  # global index within test set
        global_idx = 0

        start = time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                batch_device = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch_device)
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()
                labels = batch["labels"].cpu().tolist()

                all_preds.extend(preds)
                all_labels.extend(labels)
                all_indices.extend(range(global_idx, global_idx + len(labels)))
                global_idx += len(labels)

        eval_time = time.perf_counter() - start

        # sklearn metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        per_class = classification_report(
            all_labels,
            all_preds,
            target_names=self._label_names,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(all_labels, all_preds).tolist()

        misclassified = _collect_misclassified(
            all_indices, all_labels, all_preds, max_examples=_MAX_MISCLASSIFIED
        )

        return EvaluationResult(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            per_class_report=per_class,
            confusion_matrix=cm,
            misclassified_examples=misclassified,
            evaluation_time_seconds=eval_time,
            num_test_samples=len(all_labels),
            label_names=self._label_names,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_misclassified(
    indices: list[int],
    labels: list[int],
    preds: list[int],
    max_examples: int,
) -> list[dict]:
    """
    Return up to `max_examples` misclassified entries.

    Each entry: {"idx": int, "true_label": int, "pred_label": int}
    Text is not stored here — indices let callers look up the raw
    dataset if needed, keeping this structure lightweight.
    """
    result = []
    for idx, true, pred in zip(indices, labels, preds, strict=False):
        if true != pred:
            result.append({"idx": idx, "true_label": true, "pred_label": pred})
            if len(result) >= max_examples:
                break
    return result
