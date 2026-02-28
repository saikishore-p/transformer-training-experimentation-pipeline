from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.orchestration import load_sweep_configs, run_sweep
from pipeline.registry import ModelRegistry
from pipeline.reporting import generate_summary_report, print_summary_table, save_summary_json
from rich.console import Console

_console = Console()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a parallel sweep of experiments via Ray.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Path to sweep YAML file (list of experiment configs).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum parallel Ray workers. Default: 2.",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to shared registry.json. Default: checkpoints/registry.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write sweep_results.json. Default: sweep_results.json in CWD.",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=None,
        help="Optional base YAML config to merge sweep entries over.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        _console.print(f"[red]Sweep file not found: {sweep_path}[/red]")
        sys.exit(1)

    registry_path = args.registry or "checkpoints/registry.json"
    output_path = args.output or "sweep_results.json"

    base_config = None
    if args.base_config:
        from pipeline.config import load_config  # noqa: PLC0415
        base_path = Path(args.base_config)
        if not base_path.exists():
            _console.print(f"[red]Base config not found: {base_path}[/red]")
            sys.exit(1)
        base_config = load_config(base_path)

    _console.print(f"\n[bold cyan]Loading sweep: {sweep_path}[/bold cyan]")
    try:
        configs = load_sweep_configs(str(sweep_path), base_config=base_config)
    except Exception as exc:
        _console.print(f"[red]Failed to load sweep config: {exc}[/red]")
        sys.exit(1)

    _console.print(
        f"  {len(configs)} experiment(s) loaded  |  "
        f"max_concurrent={args.max_concurrent}  |  "
        f"registry={registry_path}"
    )

    results = run_sweep(
        configs=configs,
        registry_path=registry_path,
        max_concurrent=args.max_concurrent,
    )

    _save_results(results, output_path)

    registry = ModelRegistry(registry_path)
    report = generate_summary_report(registry)
    print_summary_table(report)
    save_summary_json(report, output_path.replace(".json", "_report.json"))

    failed = [r for r in results if r.get("status") != "completed"]
    if failed:
        _console.print(
            f"\n[red]{len(failed)}/{len(results)} run(s) failed. "
            f"Check error fields in {output_path}.[/red]"
        )
        sys.exit(1)  # failure

    any_regression = any(r.get("regression_detected", False) for r in results)
    if any_regression:
        _console.print(
            "\n[yellow]Regression detected in one or more runs. "
            "Review the summary table for details.[/yellow]"
        )
        sys.exit(2)  # regression detected

    _console.print(f"\n[green]All {len(results)} run(s) completed successfully.[/green]")


def _save_results(results: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    _console.print(f"[dim]Raw results saved to {path}[/dim]")


if __name__ == "__main__":
    main()
