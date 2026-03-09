"""
Core pipeline orchestration logic.

`run_pipeline()` is the single function that executes one full experiment:
  data → model → train → evaluate → track → registry → regression check

Both the CLI script (run_experiment.py) and the Ray sweep runner
(run_sweep.py) call this function — keeping the logic in one place.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from pipeline.config import PipelineConfig
from pipeline.data import (
    get_data_metadata,
    get_label_names,
    get_num_labels,
    load_dataset_splits,
    tokenize_splits,
)
from pipeline.evaluation import Evaluator
from pipeline.models import create_model
from pipeline.monitoring import RegressionDetector
from pipeline.registry import ModelRegistry
from pipeline.tracking import MLflowTracker
from pipeline.training import CheckpointSaver, EarlyStopping, Trainer
from pipeline.training.utils import get_device

_console = Console()


def run_pipeline(config: PipelineConfig, registry_path: str | None = None) -> dict:
    """
    Execute one full training + evaluation run from a PipelineConfig.

    Steps:
        1. Load & preprocess dataset
        2. Build model
        3. Open MLflow run
        4. Log all params
        5. Train (with optional early stopping + checkpointing)
        6. Evaluate on test set
        7. Log eval metrics & artifacts (eval report, config snapshot)
        8. Register in ModelRegistry
        9. Run regression detection
       10. Log regression tags
       11. Print summary to console

    Args:
        config:        Fully-validated PipelineConfig.
        registry_path: Override path to registry.json.
                       Defaults to {config.experiment.output_dir}/registry.json.

    Returns:
        Summary dict with run_id, experiment_name, test_accuracy, regression_detected.
    """
    _print_header(config)

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    _console.print("[bold cyan]▶ Loading dataset…[/bold cyan]")
    splits = load_dataset_splits(config.data, seed=config.training.seed)
    num_labels = get_num_labels(splits)
    label_names = get_label_names(splits)
    data_meta = get_data_metadata(splits, config.data)

    _console.print(
        f"  train={data_meta['num_train']:,}  "
        f"val={data_meta['num_val']:,}  "
        f"test={data_meta['num_test']:,}  "
        f"labels={num_labels}"
    )

    # ------------------------------------------------------------------ #
    # 2. Model
    # ------------------------------------------------------------------ #
    _console.print("[bold cyan]▶ Building model…[/bold cyan]")
    model_wrapper = create_model(config.model, num_labels=num_labels)
    device = get_device()
    _console.print(
        f"  {config.model.name}  |  "
        f"params={model_wrapper.get_num_parameters():,}  |  "
        f"device={device}"
    )

    # ------------------------------------------------------------------ #
    # 3–7. Train + Evaluate inside MLflow run
    # ------------------------------------------------------------------ #
    registry_path = registry_path or f"{config.experiment.output_dir}/registry.json"
    registry = ModelRegistry(registry_path)

    with MLflowTracker(config.experiment, run_name=config.experiment.name) as tracker:
        run_id = tracker.run_id

        # ---- Params ----
        params = {**config.to_flat_dict(), **data_meta}
        params["model.num_parameters"] = model_wrapper.get_num_parameters()
        params["device"] = str(device)
        tracker.log_params(params)

        # ---- Train ----
        _console.print("[bold cyan]▶ Training…[/bold cyan]")
        tokenized = tokenize_splits(splits, model_wrapper.get_tokenizer(), config.data)
        callbacks = _build_callbacks(config)
        trainer = Trainer(
            model_wrapper=model_wrapper,
            train_dataset=tokenized["train"],
            val_dataset=tokenized["validation"],
            config=config.training,
            tracker=tracker,
            callbacks=callbacks,
        )
        training_result = trainer.train()

        _console.print(
            f"  best epoch={training_result.best_epoch}  "
            f"val_acc={training_result.best_val_accuracy:.4f}  "
            f"time={training_result.total_time_seconds:.1f}s"
        )

        # ---- Evaluate ----
        _console.print("[bold cyan]▶ Evaluating on test set…[/bold cyan]")
        evaluator = Evaluator(
            model_wrapper=model_wrapper,
            test_dataset=tokenized["test"],
            label_names=label_names,
        )
        eval_result = evaluator.evaluate()

        _console.print(
            f"  accuracy={eval_result.accuracy:.4f}  "
            f"f1={eval_result.f1:.4f}  "
            f"time={eval_result.evaluation_time_seconds:.2f}s"
        )

        # ---- Log eval metrics ----
        tracker.log_metrics(eval_result.summary_metrics())
        tracker.log_metrics(
            {
                "best_val_accuracy": training_result.best_val_accuracy,
                "best_epoch": float(training_result.best_epoch),
            }
        )

        # ---- Log artifacts ----
        _log_artifacts(tracker, config, eval_result, training_result)

        # ---- Register ----
        _console.print("[bold cyan]▶ Registering model…[/bold cyan]")
        ckpt_path = training_result.checkpoint_path or "no_checkpoint"
        registry.register(
            run_id=run_id,
            experiment_name=config.experiment.name,
            model_name=config.model.name,
            checkpoint_path=ckpt_path,
            test_accuracy=eval_result.accuracy,
            val_accuracy=training_result.best_val_accuracy,
            config_snapshot=config.to_flat_dict(),
            test_f1=eval_result.f1,
            total_training_time_seconds=training_result.total_time_seconds,
            num_parameters=model_wrapper.get_num_parameters(),
        )

        # ---- Regression detection ----
        _console.print("[bold cyan]▶ Checking for regression…[/bold cyan]")
        detector = RegressionDetector(registry)
        regression_result = detector.check(eval_result.accuracy, run_id=run_id)
        regression_result.print_summary()

        tracker.set_tags(regression_result.to_mlflow_tags())
        tracker.log_dict(regression_result.to_dict(), "regression_result.json")

    # ---- Final summary ----
    _print_summary(config, run_id, eval_result, training_result, regression_result)

    return {
        "run_id": run_id,
        "experiment_name": config.experiment.name,
        "test_accuracy": eval_result.accuracy,
        "test_f1": eval_result.f1,
        "regression_detected": regression_result.regression_detected,
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_callbacks(config: PipelineConfig) -> list:
    callbacks = []
    if config.training.early_stopping_patience is not None:
        callbacks.append(EarlyStopping(patience=config.training.early_stopping_patience))
    callbacks.append(
        CheckpointSaver(
            output_dir=config.experiment.output_dir,
            experiment_name=config.experiment.name,
        )
    )
    return callbacks


def _log_artifacts(tracker, config, eval_result, training_result) -> None:
    """Write eval report + config snapshot to temp files and upload to MLflow."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        eval_path = tmp_path / "eval_report.json"
        eval_result.to_json(str(eval_path))
        tracker.log_artifact(str(eval_path))

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config.to_flat_dict(), indent=2))
        tracker.log_artifact(str(config_path))

        metrics_path = tmp_path / "training_metrics.json"
        metrics_path.write_text(
            json.dumps(training_result.per_epoch_metrics, indent=2, default=str)
        )
        tracker.log_artifact(str(metrics_path))


def _print_header(config: PipelineConfig) -> None:
    _console.print(Rule(f"[bold]Experiment: {config.experiment.name}[/bold]"))
    _console.print(
        f"  model=[cyan]{config.model.name}[/cyan]  "
        f"dataset=[cyan]{config.data.dataset_name}[/cyan]  "
        f"lr={config.training.learning_rate}  "
        f"epochs={config.training.epochs}"
    )


def _print_summary(config, run_id, eval_result, training_result, regression_result) -> None:
    _console.print(Rule("[bold]Run Summary[/bold]"))
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Run ID", f"[dim]{run_id}[/dim]")
    table.add_row("Experiment", config.experiment.name)
    table.add_row("Model", config.model.name)
    table.add_row("Test Accuracy", f"[bold green]{eval_result.accuracy:.4f}[/bold green]")
    table.add_row("Test F1", f"{eval_result.f1:.4f}")
    table.add_row("Best Epoch", str(training_result.best_epoch))
    table.add_row("Training Time", f"{training_result.total_time_seconds:.1f}s")
    table.add_row(
        "Regression",
        "[bold red]YES[/bold red]"
        if regression_result.regression_detected
        else "[bold green]NO[/bold green]",
    )
    _console.print(table)
    _console.print(
        f"\n[dim]MLflow UI: mlflow ui --backend-store-uri {config.experiment.mlflow_tracking_uri}[/dim]"
    )
