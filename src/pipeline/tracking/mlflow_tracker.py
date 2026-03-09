"""
MLflow experiment tracking integration.

MLflowTracker is a context manager that wraps a single MLflow run.
Every pipeline run opens one tracker, logs everything through it,
and closes cleanly on exit — even if training raises an exception.

Usage:
    with MLflowTracker(config.experiment) as tracker:
        tracker.log_params(config.to_flat_dict())
        trainer = Trainer(..., tracker=tracker)
        result = trainer.train()
        tracker.log_metrics({"test_accuracy": 0.92})
        tracker.log_artifact("eval_report.json")
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import mlflow

from pipeline.config import ExperimentConfig


class MLflowTracker:
    """
    Thin, opinionated wrapper around a single MLflow run.

    Responsibilities:
      - Set the tracking URI and experiment name from ExperimentConfig
      - Start / end a run (as a context manager)
      - Provide simple helpers so callers never touch mlflow directly

    The Trainer and other pipeline stages only depend on the method
    signatures here — not on MLflow internals — making the tracker easy
    to stub in tests.
    """

    def __init__(self, config: ExperimentConfig, run_name: str | None = None) -> None:
        self._config = config
        self._run_name = run_name or config.name
        self._run: mlflow.ActiveRun | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> MLflowTracker:
        mlflow.set_tracking_uri(self._config.mlflow_tracking_uri)
        mlflow.set_experiment(self._config.mlflow_experiment_name)
        self._run = mlflow.start_run(run_name=self._run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Always end the run, even on exception — avoids orphaned active runs
        status = "FAILED" if exc_type is not None else "FINISHED"
        mlflow.end_run(status=status)
        self._run = None
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log a flat dict of parameters.
        Values are cast to str; None values are logged as "None".
        MLflow has a 500-char limit per param value — long values are truncated.
        """
        safe = {k: str(v)[:500] for k, v in params.items()}
        mlflow.log_params(safe)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single scalar metric, optionally at a given step (epoch)."""
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple scalar metrics at once."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str) -> None:
        """Upload a local file or directory to the run's artifact store."""
        mlflow.log_artifact(local_path)

    def log_artifact_dir(self, local_dir: str, artifact_path: str | None = None) -> None:
        """Upload an entire local directory to the run's artifact store."""
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    def log_dict(self, data: dict, artifact_filename: str) -> None:
        """
        Serialize `data` to JSON and upload as an artifact.
        Useful for eval reports, regression results, config snapshots.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="mlflow_"
        ) as f:
            json.dump(data, f, indent=2, default=str)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="")
        # Rename artifact inside MLflow by logging with the desired name
        # (MLflow doesn't support rename; re-log with target name)
        import os
        import shutil

        target = Path(tmp_path).parent / artifact_filename
        shutil.copy(tmp_path, target)
        mlflow.log_artifact(str(target))
        os.unlink(tmp_path)
        os.unlink(str(target))

    def set_tag(self, key: str, value: str) -> None:
        """Set a string tag on the current run (e.g. 'regression_detected': 'true')."""
        mlflow.set_tag(key, value)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set multiple string tags at once."""
        mlflow.set_tags(tags)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_run_id(self) -> str:
        """Return the MLflow run ID for the active run."""
        if self._run is None:
            raise RuntimeError("MLflowTracker is not inside a 'with' block.")
        return self._run.info.run_id

    @property
    def run_id(self) -> str:
        return self.get_run_id()

    @property
    def experiment_name(self) -> str:
        return self._config.mlflow_experiment_name

    def __repr__(self) -> str:
        run_id = self._run.info.run_id if self._run else "not started"
        return f"MLflowTracker(experiment={self.experiment_name!r}, run_id={run_id!r})"
