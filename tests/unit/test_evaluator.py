"""
Unit tests for the evaluation layer.

Uses a FakeModel with fully controllable predictions — no real model
downloads. Tests verify metric math, JSON serialisation, and edge cases
like a perfect classifier, all-wrong classifier, and class imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from unittest.mock import MagicMock

from datasets import Dataset
import pytest
import torch
import torch.nn as nn

from pipeline.evaluation import EvaluationResult, Evaluator

# ---------------------------------------------------------------------------
# Fake model that returns pre-specified predictions
# ---------------------------------------------------------------------------


@dataclass
class _FakeOutput:
    loss: torch.Tensor
    logits: torch.Tensor


class _FixedPredModel(nn.Module):
    """
    Returns logits that make the model predict a fixed label for every sample.
    Set `override_preds` to a list to return specific per-sample predictions.
    """

    def __init__(self, num_labels: int = 4, fixed_pred: int = 0):
        super().__init__()
        self.num_labels = num_labels
        self.fixed_pred = fixed_pred
        self.override_preds: list[int] | None = None  # set externally for fine-grained control
        self._call_count = 0

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batch_size = input_ids.shape[0]

        if self.override_preds is not None:
            start = self._call_count
            batch_preds = self.override_preds[start : start + batch_size]
            self._call_count += batch_size
        else:
            batch_preds = [self.fixed_pred] * batch_size

        # Build one-hot-ish logits so argmax == batch_preds
        logits = torch.zeros(batch_size, self.num_labels)
        for i, p in enumerate(batch_preds):
            logits[i, p] = 10.0  # large value → argmax = p

        loss = torch.tensor(0.0)
        return _FakeOutput(loss=loss, logits=logits)


def _make_wrapper(model: nn.Module) -> MagicMock:
    wrapper = MagicMock()
    wrapper.get_model.return_value = model
    return wrapper


def _make_dataset(labels: list[int], seq_len: int = 8) -> Dataset:
    n = len(labels)
    return Dataset.from_dict(
        {
            "input_ids": [[1] * seq_len] * n,
            "attention_mask": [[1] * seq_len] * n,
            "labels": labels,
        }
    ).with_format("torch")


# ---------------------------------------------------------------------------
# EvaluationResult tests
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def _make_result(self, **overrides) -> EvaluationResult:
        defaults = dict(
            accuracy=0.9,
            precision=0.88,
            recall=0.87,
            f1=0.875,
            per_class_report={"0": {"precision": 0.9, "recall": 0.9, "f1-score": 0.9}},
            confusion_matrix=[[9, 1], [0, 10]],
            misclassified_examples=[{"idx": 3, "true_label": 0, "pred_label": 1}],
            evaluation_time_seconds=1.23,
            num_test_samples=20,
        )
        defaults.update(overrides)
        return EvaluationResult(**defaults)

    def test_to_dict_has_all_keys(self):
        result = self._make_result()
        d = result.to_dict()
        for key in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "per_class_report",
            "confusion_matrix",
            "misclassified_examples",
            "evaluation_time_seconds",
            "num_test_samples",
            "label_names",
        ):
            assert key in d

    def test_to_dict_values_match(self):
        result = self._make_result(accuracy=0.75)
        assert result.to_dict()["accuracy"] == pytest.approx(0.75)

    def test_to_json_creates_valid_file(self, tmp_path):
        result = self._make_result()
        path = str(tmp_path / "eval.json")
        result.to_json(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["accuracy"] == pytest.approx(0.9)
        assert loaded["num_test_samples"] == 20

    def test_summary_metrics_keys(self):
        result = self._make_result()
        summary = result.summary_metrics()
        for key in (
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "evaluation_time_seconds",
        ):
            assert key in summary

    def test_summary_metrics_values(self):
        result = self._make_result(accuracy=0.8, f1=0.79)
        summary = result.summary_metrics()
        assert summary["test_accuracy"] == pytest.approx(0.8)
        assert summary["test_f1"] == pytest.approx(0.79)

    def test_label_names_default_none(self):
        result = self._make_result()
        assert result.label_names is None

    def test_label_names_stored(self):
        result = self._make_result(label_names=["A", "B", "C"])
        assert result.label_names == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Evaluator — perfect classifier
# ---------------------------------------------------------------------------


class TestEvaluatorPerfectClassifier:
    """Model always predicts the correct label → accuracy = 1.0"""

    @pytest.fixture
    def result(self):
        num_labels = 4
        # Labels cycle 0,1,2,3,0,1,2,3,...  model predicts same
        labels = [i % num_labels for i in range(40)]
        model = _FixedPredModel(num_labels=num_labels)
        model.override_preds = labels[:]
        evaluator = Evaluator(_make_wrapper(model), _make_dataset(labels), batch_size=10)
        return evaluator.evaluate()

    def test_accuracy_is_one(self, result):
        assert result.accuracy == pytest.approx(1.0)

    def test_f1_is_one(self, result):
        assert result.f1 == pytest.approx(1.0)

    def test_no_misclassified_examples(self, result):
        assert result.misclassified_examples == []

    def test_confusion_matrix_is_diagonal(self, result):
        cm = result.confusion_matrix
        for i, row in enumerate(cm):
            for j, val in enumerate(row):
                if i == j:
                    assert val > 0
                else:
                    assert val == 0


# ---------------------------------------------------------------------------
# Evaluator — all-wrong classifier
# ---------------------------------------------------------------------------


class TestEvaluatorAllWrong:
    """
    Model always predicts label 0 for every sample.
    True labels are 1,2,3 only → accuracy = 0.
    """

    @pytest.fixture
    def result(self):
        labels = [1, 2, 3] * 10  # never label 0
        model = _FixedPredModel(num_labels=4, fixed_pred=0)
        evaluator = Evaluator(_make_wrapper(model), _make_dataset(labels), batch_size=10)
        return evaluator.evaluate()

    def test_accuracy_is_zero(self, result):
        assert result.accuracy == pytest.approx(0.0)

    def test_has_misclassified_examples(self, result):
        assert len(result.misclassified_examples) > 0

    def test_misclassified_capped_at_max(self, result):
        assert len(result.misclassified_examples) <= 20

    def test_misclassified_entry_structure(self, result):
        entry = result.misclassified_examples[0]
        assert "idx" in entry
        assert "true_label" in entry
        assert "pred_label" in entry

    def test_num_test_samples_correct(self, result):
        assert result.num_test_samples == 30


# ---------------------------------------------------------------------------
# Evaluator — controlled partial accuracy
# ---------------------------------------------------------------------------


class TestEvaluatorPartialAccuracy:
    """First half correct, second half all-wrong → accuracy = 0.5"""

    def test_accuracy_is_half(self):
        n = 20
        true_labels = [0] * n
        # first 10 correct (pred=0), next 10 wrong (pred=1)
        preds = [0] * 10 + [1] * 10
        model = _FixedPredModel(num_labels=2)
        model.override_preds = preds[:]
        evaluator = Evaluator(_make_wrapper(model), _make_dataset(true_labels), batch_size=5)
        result = evaluator.evaluate()
        assert result.accuracy == pytest.approx(0.5)

    def test_eval_time_is_positive(self):
        labels = [0] * 10
        model = _FixedPredModel(num_labels=2, fixed_pred=0)
        evaluator = Evaluator(_make_wrapper(model), _make_dataset(labels))
        result = evaluator.evaluate()
        assert result.evaluation_time_seconds > 0


# ---------------------------------------------------------------------------
# Evaluator — label names passed through
# ---------------------------------------------------------------------------


class TestEvaluatorLabelNames:
    def test_label_names_in_result(self):
        labels = [0, 1, 2, 3] * 5
        model = _FixedPredModel(num_labels=4)
        model.override_preds = labels[:]
        names = ["World", "Sports", "Business", "SciTech"]
        evaluator = Evaluator(_make_wrapper(model), _make_dataset(labels), label_names=names)
        result = evaluator.evaluate()
        assert result.label_names == names

    def test_per_class_report_uses_label_names(self):
        labels = [0, 1] * 10
        model = _FixedPredModel(num_labels=2)
        model.override_preds = labels[:]
        evaluator = Evaluator(
            _make_wrapper(model), _make_dataset(labels), label_names=["ham", "spam"]
        )
        result = evaluator.evaluate()
        assert "ham" in result.per_class_report
        assert "spam" in result.per_class_report


# ---------------------------------------------------------------------------
# Evaluator — confusion matrix shape
# ---------------------------------------------------------------------------


class TestEvaluatorConfusionMatrix:
    def test_confusion_matrix_shape(self):
        num_labels = 4
        labels = list(range(num_labels)) * 5
        model = _FixedPredModel(num_labels=num_labels)
        model.override_preds = labels[:]
        evaluator = Evaluator(_make_wrapper(model), _make_dataset(labels))
        result = evaluator.evaluate()
        assert len(result.confusion_matrix) == num_labels
        assert all(len(row) == num_labels for row in result.confusion_matrix)

    def test_confusion_matrix_row_sums_equal_class_counts(self):
        labels = [0, 0, 1, 1, 2, 2]  # 2 samples per class
        preds = [0, 1, 1, 1, 0, 2]  # some wrong
        model = _FixedPredModel(num_labels=3)
        model.override_preds = preds[:]
        evaluator = Evaluator(_make_wrapper(model), _make_dataset(labels))
        result = evaluator.evaluate()
        # Row sums = number of true samples per class
        for row in result.confusion_matrix:
            assert sum(row) == 2
