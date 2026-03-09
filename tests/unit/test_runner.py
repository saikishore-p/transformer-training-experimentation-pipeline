"""
Unit tests for pipeline/runner.py.

All heavy dependencies (HF datasets, model downloads, MLflow, Ray) are
mocked at the `pipeline.runner` module level so these tests run instantly
with no network access.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from pipeline.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
)
from pipeline.evaluation.evaluator import EvaluationResult
from pipeline.monitoring.regression_detector import RegressionResult
from pipeline.runner import _build_callbacks, _log_artifacts, run_pipeline
from pipeline.training.trainer import TrainingResult

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    early_stopping_patience: int | None = None,
    output_dir: str = "/tmp/test_runs",
) -> PipelineConfig:
    return PipelineConfig(
        experiment=ExperimentConfig(
            name="test_exp",
            output_dir=output_dir,
            mlflow_tracking_uri="file:///tmp/mlruns_test",
        ),
        model=ModelConfig(name="distilbert-base-uncased"),
        data=DataConfig(
            dataset_name="ag_news",
            subsample_train=50,
            subsample_val=10,
        ),
        training=TrainingConfig(
            epochs=1,
            learning_rate=2e-5,
            early_stopping_patience=early_stopping_patience,
        ),
    )


def _make_training_result(checkpoint_path: str | None = "/tmp/ckpt.pt") -> TrainingResult:
    return TrainingResult(
        best_val_loss=0.5,
        best_val_accuracy=0.82,
        best_epoch=1,
        total_time_seconds=5.0,
        samples_per_second=100.0,
        checkpoint_path=checkpoint_path,
        per_epoch_metrics=[{"epoch": 1, "train_loss": 0.6, "val_accuracy": 0.82}],
    )


def _make_eval_result() -> EvaluationResult:
    return EvaluationResult(
        accuracy=0.85,
        precision=0.85,
        recall=0.85,
        f1=0.84,
        per_class_report={},
        confusion_matrix=[[4, 0], [0, 4]],
        misclassified_examples=[],
        evaluation_time_seconds=0.3,
        num_test_samples=10,
        label_names=["World", "Sports"],
    )


def _make_regression_result(*, regression_detected: bool = False) -> RegressionResult:
    return RegressionResult(
        regression_detected=regression_detected,
        current_accuracy=0.85,
        best_accuracy=0.82,
        best_run_id="prev_run",
        delta=0.03,
        threshold=0.02,
    )


# Patch context: replace every heavy dependency in `pipeline.runner`
_PATCH_TARGETS = {
    "load_dataset_splits": "pipeline.runner.load_dataset_splits",
    "get_num_labels": "pipeline.runner.get_num_labels",
    "get_label_names": "pipeline.runner.get_label_names",
    "get_data_metadata": "pipeline.runner.get_data_metadata",
    "create_model": "pipeline.runner.create_model",
    "get_device": "pipeline.runner.get_device",
    "ModelRegistry": "pipeline.runner.ModelRegistry",
    "MLflowTracker": "pipeline.runner.MLflowTracker",
    "tokenize_splits": "pipeline.runner.tokenize_splits",
    "Trainer": "pipeline.runner.Trainer",
    "Evaluator": "pipeline.runner.Evaluator",
    "RegressionDetector": "pipeline.runner.RegressionDetector",
}


class _MockedPipeline:
    """Context manager that patches all runner dependencies."""

    def __init__(
        self,
        *,
        regression_detected: bool = False,
        checkpoint_path: str | None = "/tmp/ckpt.pt",
    ):
        self._patches = {}
        self._mocks: dict[str, MagicMock] = {}
        self._regression_detected = regression_detected
        self._checkpoint_path = checkpoint_path

    def __enter__(self):
        for name, target in _PATCH_TARGETS.items():
            p = patch(target)
            m = p.start()
            self._patches[name] = p
            self._mocks[name] = m

        # Wire return values
        training_result = _make_training_result(self._checkpoint_path)
        eval_result = _make_eval_result()
        regression_result = _make_regression_result(regression_detected=self._regression_detected)

        mock_splits = MagicMock()
        self._mocks["load_dataset_splits"].return_value = mock_splits
        self._mocks["get_num_labels"].return_value = 4
        self._mocks["get_label_names"].return_value = ["World", "Sports", "Business", "Sci/Tech"]
        self._mocks["get_data_metadata"].return_value = {
            "dataset_name": "ag_news",
            "dataset_config": "default",
            "num_train": 50,
            "num_val": 10,
            "num_test": 10,
            "label_distribution_train": '{"0": 12, "1": 13}',
            "dataset_fingerprint": "abcd1234",
        }

        mock_model = MagicMock()
        mock_model.get_num_parameters.return_value = 66_000_000
        mock_model.get_tokenizer.return_value = MagicMock()
        self._mocks["create_model"].return_value = mock_model

        self._mocks["get_device"].return_value = "cpu"

        mock_registry = MagicMock()
        self._mocks["ModelRegistry"].return_value = mock_registry

        mock_tracker = MagicMock()
        mock_tracker.__enter__ = MagicMock(return_value=mock_tracker)
        mock_tracker.__exit__ = MagicMock(return_value=False)
        mock_tracker.run_id = "test_run_001"
        self._mocks["MLflowTracker"].return_value = mock_tracker

        mock_tokenized = {"train": MagicMock(), "validation": MagicMock(), "test": MagicMock()}
        self._mocks["tokenize_splits"].return_value = mock_tokenized

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = training_result
        self._mocks["Trainer"].return_value = mock_trainer

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = eval_result
        self._mocks["Evaluator"].return_value = mock_evaluator

        mock_detector = MagicMock()
        mock_detector.check.return_value = regression_result
        self._mocks["RegressionDetector"].return_value = mock_detector

        self._training_result = training_result
        self._eval_result = eval_result
        self._regression_result = regression_result
        return self._mocks

    def __exit__(self, *_):
        for p in self._patches.values():
            p.stop()


# ---------------------------------------------------------------------------
# Return value tests
# ---------------------------------------------------------------------------


class TestRunPipelineReturnValue:
    def test_returns_dict(self):
        with _MockedPipeline():
            result = run_pipeline(_make_config())
        assert isinstance(result, dict)

    def test_run_id_in_result(self):
        with _MockedPipeline():
            result = run_pipeline(_make_config())
        assert result["run_id"] == "test_run_001"

    def test_experiment_name_in_result(self):
        with _MockedPipeline():
            result = run_pipeline(_make_config())
        assert result["experiment_name"] == "test_exp"

    def test_test_accuracy_in_result(self):
        with _MockedPipeline():
            result = run_pipeline(_make_config())
        assert result["test_accuracy"] == pytest.approx(0.85)

    def test_test_f1_in_result(self):
        with _MockedPipeline():
            result = run_pipeline(_make_config())
        assert result["test_f1"] == pytest.approx(0.84)

    def test_status_is_completed(self):
        with _MockedPipeline():
            result = run_pipeline(_make_config())
        assert result["status"] == "completed"

    def test_regression_detected_false_when_clean(self):
        with _MockedPipeline(regression_detected=False):
            result = run_pipeline(_make_config())
        assert result["regression_detected"] is False

    def test_regression_detected_true_when_flagged(self):
        with _MockedPipeline(regression_detected=True):
            result = run_pipeline(_make_config())
        assert result["regression_detected"] is True


# ---------------------------------------------------------------------------
# Component interaction tests
# ---------------------------------------------------------------------------


class TestRunPipelineInteractions:
    def test_load_dataset_splits_called(self):
        with _MockedPipeline() as mocks:
            run_pipeline(_make_config())
        mocks["load_dataset_splits"].assert_called_once()

    def test_create_model_called_with_num_labels(self):
        with _MockedPipeline() as mocks:
            run_pipeline(_make_config())
        mocks["create_model"].assert_called_once()
        _, kwargs = mocks["create_model"].call_args
        assert kwargs.get("num_labels") == 4

    def test_trainer_train_called(self):
        with _MockedPipeline() as mocks:
            run_pipeline(_make_config())
        mocks["Trainer"].return_value.train.assert_called_once()

    def test_evaluator_evaluate_called(self):
        with _MockedPipeline() as mocks:
            run_pipeline(_make_config())
        mocks["Evaluator"].return_value.evaluate.assert_called_once()

    def test_registry_register_called(self):
        with _MockedPipeline() as mocks:
            run_pipeline(_make_config())
        mocks["ModelRegistry"].return_value.register.assert_called_once()

    def test_regression_detector_check_called(self):
        with _MockedPipeline() as mocks:
            run_pipeline(_make_config())
        mocks["RegressionDetector"].return_value.check.assert_called_once()

    def test_tracker_log_params_called(self):
        with _MockedPipeline() as mocks:
            run_pipeline(_make_config())
        mocks["MLflowTracker"].return_value.log_params.assert_called_once()

    def test_tracker_log_metrics_called(self):
        with _MockedPipeline() as mocks:
            run_pipeline(_make_config())
        # Called at least twice: eval metrics + training metrics
        assert mocks["MLflowTracker"].return_value.log_metrics.call_count >= 2

    def test_tracker_set_tags_called_with_regression_tags(self):
        with _MockedPipeline() as mocks:
            run_pipeline(_make_config())
        mocks["MLflowTracker"].return_value.set_tags.assert_called_once()

    def test_registry_path_uses_output_dir_by_default(self):
        with _MockedPipeline() as mocks:
            run_pipeline(_make_config(output_dir="/tmp/my_runs"))
        call_args = mocks["ModelRegistry"].call_args
        assert "/tmp/my_runs" in str(call_args)

    def test_registry_path_override_respected(self):
        with _MockedPipeline() as mocks:
            run_pipeline(_make_config(), registry_path="/custom/registry.json")
        call_args = mocks["ModelRegistry"].call_args
        assert "/custom/registry.json" in str(call_args)


# ---------------------------------------------------------------------------
# _build_callbacks tests
# ---------------------------------------------------------------------------


class TestBuildCallbacks:
    def test_no_early_stopping_when_patience_none(self):
        from pipeline.training.callbacks import EarlyStopping  # noqa: PLC0415

        config = _make_config(early_stopping_patience=None)
        callbacks = _build_callbacks(config)
        assert not any(isinstance(c, EarlyStopping) for c in callbacks)

    def test_early_stopping_added_when_patience_set(self):
        from pipeline.training.callbacks import EarlyStopping  # noqa: PLC0415

        config = _make_config(early_stopping_patience=3)
        callbacks = _build_callbacks(config)
        es = [c for c in callbacks if isinstance(c, EarlyStopping)]
        assert len(es) == 1
        assert es[0].patience == 3

    def test_checkpoint_saver_always_added(self):
        from pipeline.training.callbacks import CheckpointSaver  # noqa: PLC0415

        config = _make_config()
        callbacks = _build_callbacks(config)
        assert any(isinstance(c, CheckpointSaver) for c in callbacks)

    def test_callbacks_list_length_without_early_stopping(self):
        config = _make_config(early_stopping_patience=None)
        callbacks = _build_callbacks(config)
        assert len(callbacks) == 1  # only CheckpointSaver

    def test_callbacks_list_length_with_early_stopping(self):
        config = _make_config(early_stopping_patience=2)
        callbacks = _build_callbacks(config)
        assert len(callbacks) == 2  # EarlyStopping + CheckpointSaver


# ---------------------------------------------------------------------------
# _log_artifacts tests
# ---------------------------------------------------------------------------


class TestLogArtifacts:
    def test_log_artifact_called_three_times(self, tmp_path):
        """eval_report, config.json, training_metrics.json."""
        mock_tracker = MagicMock()
        config = _make_config(output_dir=str(tmp_path))
        eval_result = _make_eval_result()
        training_result = _make_training_result()

        _log_artifacts(mock_tracker, config, eval_result, training_result)

        assert mock_tracker.log_artifact.call_count == 3

    def test_eval_report_artifact_uploaded(self, tmp_path):
        mock_tracker = MagicMock()
        config = _make_config(output_dir=str(tmp_path))
        eval_result = _make_eval_result()
        training_result = _make_training_result()

        _log_artifacts(mock_tracker, config, eval_result, training_result)

        paths = [str(c.args[0]) for c in mock_tracker.log_artifact.call_args_list]
        assert any("eval_report" in p for p in paths)

    def test_config_snapshot_artifact_uploaded(self, tmp_path):
        mock_tracker = MagicMock()
        config = _make_config(output_dir=str(tmp_path))
        eval_result = _make_eval_result()
        training_result = _make_training_result()

        _log_artifacts(mock_tracker, config, eval_result, training_result)

        paths = [str(c.args[0]) for c in mock_tracker.log_artifact.call_args_list]
        assert any("config" in p for p in paths)

    def test_training_metrics_artifact_uploaded(self, tmp_path):
        mock_tracker = MagicMock()
        config = _make_config(output_dir=str(tmp_path))
        eval_result = _make_eval_result()
        training_result = _make_training_result()

        _log_artifacts(mock_tracker, config, eval_result, training_result)

        paths = [str(c.args[0]) for c in mock_tracker.log_artifact.call_args_list]
        assert any("training_metrics" in p for p in paths)

    def test_no_checkpoint_path_does_not_raise(self):
        """run_pipeline handles checkpoint_path=None."""
        with _MockedPipeline(checkpoint_path=None):
            result = run_pipeline(_make_config())
        assert result["status"] == "completed"
