"""
Unit tests for MLflowTracker.

Uses a real MLflow tracking server backed by a tmp_path directory.
No mocking of MLflow internals — we verify actual runs, params, and
metrics land in the store, which is what matters for pipeline correctness.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest

from pipeline.config import ExperimentConfig
from pipeline.tracking import MLflowTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, name: str = "test-run") -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        mlflow_experiment_name="test-experiment",
    )


def _get_run(tracking_uri: str, run_id: str) -> mlflow.entities.Run:
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    return client.get_run(run_id)


# ---------------------------------------------------------------------------
# Context manager behaviour
# ---------------------------------------------------------------------------


class TestMLflowTrackerContextManager:
    def test_enters_and_exits_cleanly(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config) as tracker:
            assert tracker is not None

    def test_run_id_available_inside_context(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config) as tracker:
            run_id = tracker.get_run_id()
            assert isinstance(run_id, str)
            assert len(run_id) > 0

    def test_run_id_via_property(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config) as tracker:
            assert tracker.run_id == tracker.get_run_id()

    def test_run_id_raises_outside_context(self, tmp_path):
        config = _make_config(tmp_path)
        tracker = MLflowTracker(config)
        with pytest.raises(RuntimeError, match="not inside a 'with' block"):
            _ = tracker.run_id

    def test_run_status_finished_on_clean_exit(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config) as tracker:
            run_id = tracker.run_id
        run = _get_run(str(tmp_path / "mlruns"), run_id)
        assert run.info.status == "FINISHED"

    def test_run_status_failed_on_exception(self, tmp_path):
        config = _make_config(tmp_path)
        run_id = None
        with pytest.raises(ValueError), MLflowTracker(config) as tracker:
            run_id = tracker.run_id
            raise ValueError("intentional failure")
        run = _get_run(str(tmp_path / "mlruns"), run_id)
        assert run.info.status == "FAILED"

    def test_exception_is_not_suppressed(self, tmp_path):
        config = _make_config(tmp_path)
        with pytest.raises(RuntimeError, match="boom"), MLflowTracker(config):
            raise RuntimeError("boom")

    def test_experiment_name_property(self, tmp_path):
        config = _make_config(tmp_path)
        tracker = MLflowTracker(config)
        assert tracker.experiment_name == "test-experiment"

    def test_custom_run_name(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config, run_name="my-custom-run") as tracker:
            run_id = tracker.run_id
        run = _get_run(str(tmp_path / "mlruns"), run_id)
        assert run.info.run_name == "my-custom-run"


# ---------------------------------------------------------------------------
# Parameter logging
# ---------------------------------------------------------------------------


class TestLogParams:
    def test_params_stored_in_run(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config) as tracker:
            run_id = tracker.run_id
            tracker.log_params({"lr": 2e-5, "batch_size": 16, "model": "distilbert"})
        run = _get_run(str(tmp_path / "mlruns"), run_id)
        assert run.data.params["lr"] == "2e-05"
        assert run.data.params["batch_size"] == "16"
        assert run.data.params["model"] == "distilbert"

    def test_none_values_serialised_as_string(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config) as tracker:
            run_id = tracker.run_id
            tracker.log_params({"patience": None})
        run = _get_run(str(tmp_path / "mlruns"), run_id)
        assert run.data.params["patience"] == "None"

    def test_long_value_is_truncated(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config) as tracker:
            run_id = tracker.run_id
            tracker.log_params({"long_key": "x" * 600})
        run = _get_run(str(tmp_path / "mlruns"), run_id)
        assert len(run.data.params["long_key"]) == 500


# ---------------------------------------------------------------------------
# Metric logging
# ---------------------------------------------------------------------------


class TestLogMetrics:
    def test_single_metric_stored(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config) as tracker:
            run_id = tracker.run_id
            tracker.log_metric("val_accuracy", 0.92, step=1)
        run = _get_run(str(tmp_path / "mlruns"), run_id)
        assert run.data.metrics["val_accuracy"] == pytest.approx(0.92)

    def test_multiple_metrics_stored(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config) as tracker:
            run_id = tracker.run_id
            tracker.log_metrics(
                {
                    "train_loss": 0.5,
                    "val_loss": 0.6,
                    "val_accuracy": 0.85,
                },
                step=2,
            )
        run = _get_run(str(tmp_path / "mlruns"), run_id)
        assert run.data.metrics["train_loss"] == pytest.approx(0.5)
        assert run.data.metrics["val_loss"] == pytest.approx(0.6)
        assert run.data.metrics["val_accuracy"] == pytest.approx(0.85)

    def test_metrics_across_steps(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config) as tracker:
            run_id = tracker.run_id
            for epoch in range(1, 4):
                tracker.log_metrics({"val_loss": 1.0 / epoch}, step=epoch)
        client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
        history = client.get_metric_history(run_id, "val_loss")
        assert len(history) == 3
        steps = [m.step for m in history]
        assert steps == [1, 2, 3]


# ---------------------------------------------------------------------------
# Tag logging
# ---------------------------------------------------------------------------


class TestLogTags:
    def test_single_tag_stored(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config) as tracker:
            run_id = tracker.run_id
            tracker.set_tag("regression_detected", "false")
        run = _get_run(str(tmp_path / "mlruns"), run_id)
        assert run.data.tags["regression_detected"] == "false"

    def test_multiple_tags_stored(self, tmp_path):
        config = _make_config(tmp_path)
        with MLflowTracker(config) as tracker:
            run_id = tracker.run_id
            tracker.set_tags({"model_type": "huggingface", "dataset": "ag_news"})
        run = _get_run(str(tmp_path / "mlruns"), run_id)
        assert run.data.tags["model_type"] == "huggingface"
        assert run.data.tags["dataset"] == "ag_news"


# ---------------------------------------------------------------------------
# Artifact logging
# ---------------------------------------------------------------------------


class TestLogArtifacts:
    def test_log_artifact_file(self, tmp_path):
        config = _make_config(tmp_path)
        artifact_file = tmp_path / "report.txt"
        artifact_file.write_text("hello")
        with MLflowTracker(config) as tracker:
            run_id = tracker.run_id
            tracker.log_artifact(str(artifact_file))
        client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
        artifacts = client.list_artifacts(run_id)
        names = [a.path for a in artifacts]
        assert "report.txt" in names

    def test_log_dict_creates_artifact(self, tmp_path):
        config = _make_config(tmp_path)
        data = {"accuracy": 0.91, "f1": 0.89}
        with MLflowTracker(config) as tracker:
            run_id = tracker.run_id
            tracker.log_dict(data, "eval_report.json")
        client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
        artifacts = client.list_artifacts(run_id)
        names = [a.path for a in artifacts]
        assert "eval_report.json" in names

    def test_log_dict_content_is_valid_json(self, tmp_path):
        config = _make_config(tmp_path)
        data = {"accuracy": 0.91, "labels": [0, 1, 2, 3]}
        with MLflowTracker(config) as tracker:
            run_id = tracker.run_id
            tracker.log_dict(data, "eval_report.json")
        client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
        run_info = client.get_run(run_id)
        artifact_uri = run_info.info.artifact_uri
        artifact_path = Path(artifact_uri.replace("file://", "")) / "eval_report.json"
        with open(artifact_path) as f:
            loaded = json.load(f)
        assert loaded["accuracy"] == pytest.approx(0.91)
        assert loaded["labels"] == [0, 1, 2, 3]
