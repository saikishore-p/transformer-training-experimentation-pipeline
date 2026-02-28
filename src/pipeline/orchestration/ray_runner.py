from __future__ import annotations

import time
from typing import Any
import ray
from pipeline.config import PipelineConfig

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
import yaml

_console = Console()

@ray.remote(num_cpus=2)
def _run_experiment_remote(config_dict: dict, registry_path: str) -> dict:
    try:
        from pipeline.config import PipelineConfig
        from pipeline.runner import run_pipeline

        config = PipelineConfig(**config_dict)
        return run_pipeline(config, registry_path=registry_path)

    except Exception as exc:
        return {
            "run_id": None,
            "experiment_name": config_dict.get("experiment", {}).get("name", "unknown"),
            "test_accuracy": 0.0,
            "test_f1": 0.0,
            "regression_detected": False,
            "status": "failed",
            "error": str(exc),
        }


def run_sweep(
    configs: list[PipelineConfig],
    registry_path: str,
    max_concurrent: int = 2,
) -> list[dict]:
    ray.init(ignore_reinit_error=True)
    total = len(configs)
    results: list[dict | None] = [None] * total

    _console.print(f"\n[bold]Starting sweep: {total} experiments, max_concurrent={max_concurrent}[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=_console,
    ) as progress:
        task = progress.add_task("Running experiments…", total=total)

        pending: list[tuple[int, ray.ObjectRef]] = []
        config_iter = iter(enumerate(configs))
        completed = 0

        def _submit_next() -> bool:
            try:
                idx, cfg = next(config_iter)
                ref = _run_experiment_remote.remote(cfg.model_dump(), registry_path)
                pending.append((idx, ref))
                return True
            except StopIteration:
                return False

        for _ in range(min(max_concurrent, total)):
            _submit_next()

        while pending:
            # Wait for at least one task to finish
            refs = [ref for _, ref in pending]
            ready, _ = ray.wait(refs, num_returns=1, timeout=300)

            if not ready:
                _console.print("[yellow]Warning: task timeout after 5 min[/yellow]")
                break

            # Collect finished tasks
            for done_ref in ready:
                orig_idx = next(i for i, r in pending if r == done_ref)
                pending = [(i, r) for i, r in pending if r != done_ref]

                result = ray.get(done_ref)
                results[orig_idx] = result
                completed += 1
                progress.advance(task)

                status = "✅" if result.get("status") == "completed" else "❌"
                acc = result.get("test_accuracy", 0.0)
                name = result.get("experiment_name", "?")
                _console.print(f"  {status} [{completed}/{total}] {name!r}  accuracy={acc:.4f}")

                # Submit a new task to refill the batch
                _submit_next()

    # Fill in any None entries (shouldn't happen, but be safe)
    final = [r if r is not None else {"status": "unknown"} for r in results]

    _print_sweep_summary(final)
    return final


def load_sweep_configs(sweep_path: str, base_config: PipelineConfig | None = None) -> list[PipelineConfig]:

    with open(sweep_path) as f:
        entries: list[dict] = yaml.safe_load(f)

    if not isinstance(entries, list):
        raise ValueError(
            f"Sweep YAML must be a list of config dicts, got {type(entries).__name__}"
        )

    base_dict: dict = base_config.model_dump() if base_config else {}
    configs = []
    for entry in entries:
        merged = _deep_merge(base_dict, entry)
        configs.append(PipelineConfig(**merged))
    return configs


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result

def _print_sweep_summary(results: list[dict]) -> None:
    _console.print("\n[bold]Sweep Complete — Results[/bold]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Experiment", style="dim")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Regression", justify="center")
    table.add_column("Status")

    for r in sorted(results, key=lambda x: x.get("test_accuracy", 0.0), reverse=True):
        regression = "⚠️" if r.get("regression_detected") else "✅"
        status = r.get("status", "unknown")
        acc = f"{r.get('test_accuracy', 0.0):.4f}"
        f1 = f"{r.get('test_f1', 0.0):.4f}"
        table.add_row(r.get("experiment_name", "?"), acc, f1, regression, status)

    _console.print(table)
