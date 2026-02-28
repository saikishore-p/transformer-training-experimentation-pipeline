from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from statistics import mean, stdev

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from pipeline.registry import ModelRegistry, RegistryEntry

_console = Console()


@dataclass
class RunSummary:
    """Distilled view of one registry entry for reporting purposes."""
    rank: int
    experiment_name: str
    model_name: str
    test_accuracy: float
    test_f1: float
    val_accuracy: float
    learning_rate: str       # extracted from config_snapshot
    batch_size: str
    epochs: str
    training_time_seconds: float
    num_parameters: int
    run_id: str
    regression_vs_best: float   # delta from best model (0.0 for the best itself)


@dataclass
class SummaryReport:
    """Aggregated report across all registered experiment runs."""
    total_runs: int
    best_run: RunSummary | None
    worst_run: RunSummary | None
    average_accuracy: float
    accuracy_std: float          # standard deviation (0.0 if single run)
    runs: list[RunSummary] = field(default_factory=list)   # sorted best-first

    def to_dict(self) -> dict:
        return {
            "total_runs": self.total_runs,
            "average_accuracy": self.average_accuracy,
            "accuracy_std": self.accuracy_std,
            "best_run": asdict(self.best_run) if self.best_run else None,
            "worst_run": asdict(self.worst_run) if self.worst_run else None,
            "runs": [asdict(r) for r in self.runs],
        }


def generate_summary_report(registry: ModelRegistry) -> SummaryReport:
    """
    Build a SummaryReport from all entries in the registry.

    Entries are sorted best-first by test_accuracy.
    Returns a SummaryReport with empty/zero fields if registry is empty.
    """
    entries = registry.get_all_entries()   # already sorted best-first

    if not entries:
        return SummaryReport(
            total_runs=0,
            best_run=None,
            worst_run=None,
            average_accuracy=0.0,
            accuracy_std=0.0,
            runs=[],
        )

    accuracies = [e.test_accuracy for e in entries]
    best_accuracy = accuracies[0]   # entries are best-first

    runs: list[RunSummary] = []
    for rank, entry in enumerate(entries, start=1):
        runs.append(_entry_to_run_summary(rank, entry, best_accuracy))

    avg = mean(accuracies)
    std = stdev(accuracies) if len(accuracies) > 1 else 0.0

    return SummaryReport(
        total_runs=len(entries),
        best_run=runs[0],
        worst_run=runs[-1],
        average_accuracy=avg,
        accuracy_std=std,
        runs=runs,
    )


def print_summary_table(report: SummaryReport) -> None:
    """
    Render the summary report as a rich table to stdout.

    Columns: rank, experiment, model, accuracy, F1, val_acc, lr, batch, epochs, time
    Best row is highlighted green; worst row is dim.
    """
    if report.total_runs == 0:
        _console.print("[yellow]Registry is empty — no runs to report.[/yellow]")
        return

    _console.print(Rule("[bold]Experiment Summary Report[/bold]"))

    # ---- Header stats ----
    _console.print(
        f"  Total runs: [bold]{report.total_runs}[/bold]  |  "
        f"Avg accuracy: [bold]{report.average_accuracy:.4f}[/bold]  |  "
        f"Std: {report.accuracy_std:.4f}"
    )
    if report.best_run:
        _console.print(
            f"  Best:  [bold green]{report.best_run.experiment_name}[/bold green]  "
            f"({report.best_run.test_accuracy:.4f})"
        )
    if report.worst_run and report.total_runs > 1:
        _console.print(
            f"  Worst: [dim]{report.worst_run.experiment_name}[/dim]  "
            f"({report.worst_run.test_accuracy:.4f})"
        )

    # ---- Runs table ----
    table = Table(show_header=True, header_style="bold cyan", expand=False)
    table.add_column("#",       justify="right",  style="dim",  width=3)
    table.add_column("Experiment",                              min_width=16)
    table.add_column("Model",                                   min_width=14)
    table.add_column("Acc",     justify="right",  style="bold", width=7)
    table.add_column("F1",      justify="right",               width=7)
    table.add_column("Val Acc", justify="right",               width=8)
    table.add_column("LR",      justify="right",               width=7)
    table.add_column("Batch",   justify="right",               width=6)
    table.add_column("Epochs",  justify="right",               width=7)
    table.add_column("Time (s)", justify="right",              width=9)
    table.add_column("Δ Best",  justify="right",               width=8)

    for run in report.runs:
        is_best = run.rank == 1
        is_worst = run.rank == report.total_runs and report.total_runs > 1

        row_style = "bold green" if is_best else ("dim" if is_worst else "")
        delta_str = f"{run.regression_vs_best * 100:+.2f}%" if run.rank > 1 else "—"

        # Shorten model name for display (drop org prefix if present)
        model_short = run.model_name.split("/")[-1]

        table.add_row(
            str(run.rank),
            run.experiment_name,
            model_short,
            f"{run.test_accuracy:.4f}",
            f"{run.test_f1:.4f}",
            f"{run.val_accuracy:.4f}",
            run.learning_rate,
            run.batch_size,
            run.epochs,
            f"{run.training_time_seconds:.1f}",
            delta_str,
            style=row_style,
        )

    _console.print(table)


def save_summary_json(report: SummaryReport, path: str) -> None:
    """Write the full summary report to a JSON file."""
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    _console.print(f"[dim]Summary saved to {path}[/dim]")


def _entry_to_run_summary(
    rank: int,
    entry: RegistryEntry,
    best_accuracy: float,
) -> RunSummary:
    """Convert a RegistryEntry into a RunSummary for the report."""
    snap = entry.config_snapshot or {}
    return RunSummary(
        rank=rank,
        experiment_name=entry.experiment_name,
        model_name=entry.model_name,
        test_accuracy=entry.test_accuracy,
        test_f1=entry.test_f1,
        val_accuracy=entry.val_accuracy,
        learning_rate=str(snap.get("training.learning_rate", "?")),
        batch_size=str(snap.get("training.batch_size", "?")),
        epochs=str(snap.get("training.epochs", "?")),
        training_time_seconds=entry.total_training_time_seconds,
        num_parameters=entry.num_parameters,
        run_id=entry.run_id,
        regression_vs_best=entry.test_accuracy - best_accuracy,
    )
