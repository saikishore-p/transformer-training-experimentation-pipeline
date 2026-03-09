"""
Ray-based parallel experiment orchestration.

Runs multiple experiment configs concurrently, each in its own Ray worker
process. Every worker executes the full pipeline independently and logs to
the shared MLflow store.

MPS (Apple Silicon) constraints
--------------------------------
- MPS context is per-process, not shared. Each Ray worker initialises its
  own device via get_device() inside the remote function.
- Concurrency is capped at max_concurrent (default 2) to avoid memory
  pressure when multiple models are resident simultaneously on unified memory.
- Workers are CPU-only by Ray's resource model; MPS is acquired automatically
  inside run_pipeline() via get_device().

Sweep YAML format
-----------------
A sweep config file is a YAML list of partial PipelineConfig dicts.
Each entry is deep-merged over a base config so you only specify what changes:

    - experiment:
        name: lr_sweep_1e5
      training:
        learning_rate: 1e-5
    - experiment:
        name: lr_sweep_2e5
      training:
        learning_rate: 2e-5
"""

from __future__ import annotations

import ray
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from pipeline.config import PipelineConfig

_console = Console()


# ---------------------------------------------------------------------------
# Ray remote task
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=2)
def _run_experiment_remote(config_dict: dict, registry_path: str) -> dict:
    """
    Ray remote function — executes one full pipeline run in a worker process.

    Receives config as a plain dict (must be pickle-serialisable for Ray).
    Returns a summary dict or an error dict on failure.

    Note: import run_pipeline inside the function so Ray workers initialise
    their own module state (device detection, random seeds) independently.
    """
    try:
        # Imports inside the remote function ensure each worker gets its own
        # module-level state (important for MPS context and random seeds).
        from pipeline.config import PipelineConfig  # noqa: PLC0415
        from pipeline.runner import run_pipeline  # noqa: PLC0415

        config = PipelineConfig(**config_dict)
        return run_pipeline(config, registry_path=registry_path)

    except Exception as exc:  # noqa: BLE001
        return {
            "run_id": None,
            "experiment_name": config_dict.get("experiment", {}).get("name", "unknown"),
            "test_accuracy": 0.0,
            "test_f1": 0.0,
            "regression_detected": False,
            "status": "failed",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Public sweep runner
# ---------------------------------------------------------------------------


def run_sweep(
    configs: list[PipelineConfig],
    registry_path: str,
    max_concurrent: int = 2,
) -> list[dict]:
    """
    Run a list of PipelineConfigs in parallel using Ray.

    Each config becomes one Ray task. Tasks are submitted in batches of
    `max_concurrent` to respect MPS memory limits on Apple Silicon.

    Args:
        configs:        List of fully-validated PipelineConfig objects.
        registry_path:  Shared registry.json path; all workers write to it.
                        Note: concurrent writes are safe because each worker
                        re-reads before writing (file lock via JSON overwrite).
        max_concurrent: Maximum number of parallel Ray workers. Default 2.

    Returns:
        List of result dicts (one per config), in submission order.
        Each dict has: run_id, experiment_name, test_accuracy, test_f1,
                       regression_detected, status (+ error if failed).
    """
    ray.init(ignore_reinit_error=True)

    total = len(configs)
    results: list[dict | None] = [None] * total

    _console.print(
        f"\n[bold]Starting sweep: {total} experiments, max_concurrent={max_concurrent}[/bold]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=_console,
    ) as progress:
        task = progress.add_task("Running experiments…", total=total)

        # Submit in batches to bound memory usage
        pending: list[tuple[int, ray.ObjectRef]] = []  # (original_index, ref)
        config_iter = iter(enumerate(configs))
        completed = 0

        def _submit_next() -> bool:
            """Submit the next config. Returns True if submitted, False if exhausted."""
            try:
                idx, cfg = next(config_iter)
                ref = _run_experiment_remote.remote(cfg.model_dump(), registry_path)
                pending.append((idx, ref))
                return True
            except StopIteration:
                return False

        # Fill the initial batch
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


# ---------------------------------------------------------------------------
# Sweep config loading
# ---------------------------------------------------------------------------


def load_sweep_configs(
    sweep_path: str, base_config: PipelineConfig | None = None
) -> list[PipelineConfig]:
    """
    Load a sweep YAML file (list of partial config dicts) and return
    a list of fully-validated PipelineConfig objects.

    Each entry in the sweep YAML is deep-merged over `base_config`
    (or over PipelineConfig defaults if no base is provided).

    Args:
        sweep_path:   Path to sweep YAML file.
        base_config:  Optional base config to merge over.

    Returns:
        List of PipelineConfig, one per sweep entry.
    """
    import yaml  # noqa: PLC0415

    with open(sweep_path) as f:
        entries: list[dict] = yaml.safe_load(f)

    if not isinstance(entries, list):
        raise ValueError(f"Sweep YAML must be a list of config dicts, got {type(entries).__name__}")

    base_dict: dict = base_config.model_dump() if base_config else {}
    configs = []
    for entry in entries:
        merged = _deep_merge(base_dict, entry)
        configs.append(PipelineConfig(**merged))
    return configs


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge `override` into a copy of `base`.
    Nested dicts are merged; all other values are replaced.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------


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
