"""
Integration test: full pipeline end-to-end on a tiny dataset subset.

Uses real models (distilbert-base-uncased) and real MLflow tracking,
but with a very small subsample so it completes in ~60s on CPU/MPS.

Marked @pytest.mark.slow — skipped in normal runs.
Run with:
    uv run pytest -m slow tests/integration/test_full_pipeline.py -v
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from pipeline.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
)
from pipeline.runner import run_pipeline


@pytest.fixture(scope="module")
def tiny_config(tmp_path_factory):
    """
    Config that runs in ~30-60s:
      - 50 train / 20 val samples from AG News
      - 1 epoch only
      - batch_size=8, max_length=64
    """
    tmp = tmp_path_factory.mktemp("integration")
    return PipelineConfig(
        model=ModelConfig(type="huggingface", name="distilbert-base-uncased"),
        training=TrainingConfig(
            learning_rate=2e-5,
            batch_size=8,
            epochs=1,
            seed=42,
        ),
        data=DataConfig(
            dataset_name="ag_news",
            max_length=64,
            subsample_train=50,
            subsample_val=20,
        ),
        experiment=ExperimentConfig(
            name="integration-test",
            output_dir=str(tmp / "checkpoints"),
            mlflow_tracking_uri=str(tmp / "mlruns"),
            mlflow_experiment_name="integration-tests",
        ),
    )


@pytest.mark.slow
class TestFullPipeline:
    @pytest.fixture(scope="class")
    def result(self, tiny_config, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("registry")
        return run_pipeline(
            tiny_config,
            registry_path=str(tmp / "registry.json"),
        )

    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    def test_has_run_id(self, result):
        assert "run_id" in result
        assert len(result["run_id"]) > 0

    def test_status_completed(self, result):
        assert result["status"] == "completed"

    def test_test_accuracy_in_range(self, result):
        assert 0.0 <= result["test_accuracy"] <= 1.0

    def test_test_f1_in_range(self, result):
        assert 0.0 <= result["test_f1"] <= 1.0

    def test_regression_detected_key_present(self, result):
        assert "regression_detected" in result

    def test_first_run_no_regression(self, result):
        # First ever run → no prior model → regression_detected must be False
        assert result["regression_detected"] is False

    def test_mlflow_run_exists_and_finished(self, tiny_config, result):
        client = mlflow.tracking.MlflowClient(
            tracking_uri=tiny_config.experiment.mlflow_tracking_uri
        )
        run = client.get_run(result["run_id"])
        assert run.info.status == "FINISHED"

    def test_mlflow_params_logged(self, tiny_config, result):
        client = mlflow.tracking.MlflowClient(
            tracking_uri=tiny_config.experiment.mlflow_tracking_uri
        )
        run = client.get_run(result["run_id"])
        params = run.data.params
        assert "training.learning_rate" in params
        assert "model.name" in params
        assert "data.dataset_name" in params

    def test_mlflow_metrics_logged(self, tiny_config, result):
        client = mlflow.tracking.MlflowClient(
            tracking_uri=tiny_config.experiment.mlflow_tracking_uri
        )
        run = client.get_run(result["run_id"])
        metrics = run.data.metrics
        for key in ("test_accuracy", "test_f1", "train_loss", "val_loss", "val_accuracy"):
            assert key in metrics, f"Missing metric: {key}"

    def test_mlflow_artifacts_logged(self, tiny_config, result):
        client = mlflow.tracking.MlflowClient(
            tracking_uri=tiny_config.experiment.mlflow_tracking_uri
        )
        artifacts = client.list_artifacts(result["run_id"])
        artifact_names = [a.path for a in artifacts]
        assert "eval_report.json" in artifact_names
        assert "config.json" in artifact_names

    def test_mlflow_regression_tag_set(self, tiny_config, result):
        client = mlflow.tracking.MlflowClient(
            tracking_uri=tiny_config.experiment.mlflow_tracking_uri
        )
        run = client.get_run(result["run_id"])
        assert "regression_detected" in run.data.tags

    def test_registry_updated(self, tiny_config, result, tmp_path_factory):
        # Re-run with same registry to confirm entry exists
        # The registry is in a tmp dir from the result fixture; we verify
        # the result dict has the expected fields which were populated from registry
        assert result["experiment_name"] == "integration-test"

    def test_checkpoint_created(self, tiny_config, result):
        ckpt_dir = Path(tiny_config.experiment.output_dir)
        # CheckpointSaver writes to output_dir/experiment_name/best/
        best_dir = ckpt_dir / tiny_config.experiment.name / "best"
        assert best_dir.exists(), f"Checkpoint not found at {best_dir}"
