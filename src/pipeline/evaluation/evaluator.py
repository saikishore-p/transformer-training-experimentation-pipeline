from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from torch.utils.data import DataLoader

from pipeline.models.base import ModelWrapper
from pipeline.training.utils import get_device

_MAX_MISCLASSIFIED = 20


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    per_class_report: dict
    confusion_matrix: list[list[int]]
    misclassified_examples: list[dict]
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
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary_metrics(self) -> dict[str, float]:
        return {
            "test_accuracy": self.accuracy,
            "test_precision": self.precision,
            "test_recall": self.recall,
            "test_f1": self.f1,
            "evaluation_time_seconds": self.evaluation_time_seconds,
        }


class Evaluator:

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
        all_indices: list[int] = []
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

def _collect_misclassified(
    indices: list[int],
    labels: list[int],
    preds: list[int],
    max_examples: int,
) -> list[dict]:
    result = []
    for idx, true, pred in zip(indices, labels, preds):
        if true != pred:
            result.append({"idx": idx, "true_label": true, "pred_label": pred})
            if len(result) >= max_examples:
                break
    return result
