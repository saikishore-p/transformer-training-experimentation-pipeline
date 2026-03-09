"""
Unit tests for the cross-run reporting module.

Uses a real in-memory ModelRegistry backed by tmp_path.
Tests cover empty registry, single run, multiple runs, sorting,
delta computation, and JSON serialisation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.registry import ModelRegistry
from pipeline.reporting import (
    SummaryReport,
    generate_summary_report,
    save_summary_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reg(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(str(tmp_path / "registry.json"))


def _register(
    registry: ModelRegistry,
    run_id: str,
    accuracy: float,
    experiment_name: str = "exp",
    model_name: str = "distilbert-base-uncased",
    f1: float | None = None,
    lr: float = 2e-5,
    batch: int = 16,
    epochs: int = 3,
    training_time: float = 10.0,
    num_params: int = 66_000_000,
) -> None:
    registry.register(
        run_id=run_id,
        experiment_name=experiment_name,
        model_name=model_name,
        checkpoint_path=f"ckpt/{run_id}",
        test_accuracy=accuracy,
        val_accuracy=accuracy - 0.01,
        config_snapshot={
            "training.learning_rate": str(lr),
            "training.batch_size": str(batch),
            "training.epochs": str(epochs),
        },
        test_f1=f1 if f1 is not None else accuracy - 0.01,
        total_training_time_seconds=training_time,
        num_parameters=num_params,
    )


# ---------------------------------------------------------------------------
# generate_summary_report — empty registry
# ---------------------------------------------------------------------------


class TestEmptyRegistry:
    def test_returns_summary_report(self, tmp_path):
        report = generate_summary_report(_reg(tmp_path))
        assert isinstance(report, SummaryReport)

    def test_total_runs_zero(self, tmp_path):
        report = generate_summary_report(_reg(tmp_path))
        assert report.total_runs == 0

    def test_best_run_is_none(self, tmp_path):
        report = generate_summary_report(_reg(tmp_path))
        assert report.best_run is None

    def test_worst_run_is_none(self, tmp_path):
        report = generate_summary_report(_reg(tmp_path))
        assert report.worst_run is None

    def test_average_accuracy_is_zero(self, tmp_path):
        report = generate_summary_report(_reg(tmp_path))
        assert report.average_accuracy == pytest.approx(0.0)

    def test_runs_list_is_empty(self, tmp_path):
        report = generate_summary_report(_reg(tmp_path))
        assert report.runs == []


# ---------------------------------------------------------------------------
# generate_summary_report — single run
# ---------------------------------------------------------------------------


class TestSingleRun:
    @pytest.fixture
    def report(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.88, experiment_name="baseline")
        return generate_summary_report(reg)

    def test_total_runs_one(self, report):
        assert report.total_runs == 1

    def test_best_equals_worst(self, report):
        assert report.best_run.run_id == report.worst_run.run_id == "r1"

    def test_average_equals_only_run(self, report):
        assert report.average_accuracy == pytest.approx(0.88)

    def test_std_is_zero_for_single_run(self, report):
        assert report.accuracy_std == pytest.approx(0.0)

    def test_run_has_rank_one(self, report):
        assert report.runs[0].rank == 1

    def test_regression_vs_best_is_zero_for_best(self, report):
        assert report.runs[0].regression_vs_best == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# generate_summary_report — multiple runs
# ---------------------------------------------------------------------------


class TestMultipleRuns:
    @pytest.fixture
    def report(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.75, experiment_name="low", lr=1e-5)
        _register(reg, "r2", 0.92, experiment_name="best", lr=2e-5)
        _register(reg, "r3", 0.83, experiment_name="mid", lr=3e-5)
        return generate_summary_report(reg)

    def test_total_runs(self, report):
        assert report.total_runs == 3

    def test_best_run_is_highest_accuracy(self, report):
        assert report.best_run.experiment_name == "best"
        assert report.best_run.test_accuracy == pytest.approx(0.92)

    def test_worst_run_is_lowest_accuracy(self, report):
        assert report.worst_run.experiment_name == "low"
        assert report.worst_run.test_accuracy == pytest.approx(0.75)

    def test_runs_sorted_best_first(self, report):
        accuracies = [r.test_accuracy for r in report.runs]
        assert accuracies == sorted(accuracies, reverse=True)

    def test_ranks_are_sequential(self, report):
        ranks = [r.rank for r in report.runs]
        assert ranks == [1, 2, 3]

    def test_average_accuracy_correct(self, report):
        assert report.average_accuracy == pytest.approx((0.75 + 0.92 + 0.83) / 3)

    def test_std_is_positive(self, report):
        assert report.accuracy_std > 0

    def test_best_run_delta_is_zero(self, report):
        best = next(r for r in report.runs if r.rank == 1)
        assert best.regression_vs_best == pytest.approx(0.0)

    def test_lower_runs_have_negative_delta(self, report):
        for run in report.runs[1:]:
            assert run.regression_vs_best < 0

    def test_config_snapshot_fields_extracted(self, report):
        best = report.best_run
        assert best.learning_rate == "2e-05"
        assert best.batch_size == "16"
        assert best.epochs == "3"

    def test_model_name_stored(self, report):
        assert report.best_run.model_name == "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# RunSummary — individual run fields
# ---------------------------------------------------------------------------


class TestRunSummary:
    def test_training_time_stored(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.9, training_time=123.4)
        report = generate_summary_report(reg)
        assert report.best_run.training_time_seconds == pytest.approx(123.4)

    def test_num_parameters_stored(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.9, num_params=66_955_010)
        report = generate_summary_report(reg)
        assert report.best_run.num_parameters == 66_955_010

    def test_f1_stored(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.9, f1=0.88)
        report = generate_summary_report(reg)
        assert report.best_run.test_f1 == pytest.approx(0.88)

    def test_missing_snapshot_key_returns_question_mark(self, tmp_path):
        reg = _reg(tmp_path)
        # Register with an empty config snapshot
        reg.register(
            run_id="r1",
            experiment_name="e",
            model_name="m",
            checkpoint_path="p",
            test_accuracy=0.8,
            val_accuracy=0.79,
            config_snapshot={},
        )
        report = generate_summary_report(reg)
        assert report.best_run.learning_rate == "?"


# ---------------------------------------------------------------------------
# SummaryReport.to_dict
# ---------------------------------------------------------------------------


class TestSummaryReportToDict:
    def test_to_dict_keys(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.85)
        report = generate_summary_report(reg)
        d = report.to_dict()
        for key in (
            "total_runs",
            "average_accuracy",
            "accuracy_std",
            "best_run",
            "worst_run",
            "runs",
        ):
            assert key in d

    def test_to_dict_is_json_serialisable(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.85)
        report = generate_summary_report(reg)
        serialised = json.dumps(report.to_dict())  # must not raise
        assert len(serialised) > 0

    def test_empty_report_to_dict(self, tmp_path):
        report = generate_summary_report(_reg(tmp_path))
        d = report.to_dict()
        assert d["total_runs"] == 0
        assert d["best_run"] is None


# ---------------------------------------------------------------------------
# save_summary_json
# ---------------------------------------------------------------------------


class TestSaveSummaryJson:
    def test_creates_file(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.85)
        report = generate_summary_report(reg)
        out = str(tmp_path / "summary.json")
        save_summary_json(report, out)
        assert Path(out).exists()

    def test_file_is_valid_json(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.85)
        report = generate_summary_report(reg)
        out = str(tmp_path / "summary.json")
        save_summary_json(report, out)
        with open(out) as f:
            data = json.load(f)
        assert data["total_runs"] == 1
        assert "runs" in data

    def test_multiple_runs_all_present_in_json(self, tmp_path):
        reg = _reg(tmp_path)
        for i, acc in enumerate([0.80, 0.85, 0.90]):
            _register(reg, f"r{i}", acc)
        report = generate_summary_report(reg)
        out = str(tmp_path / "summary.json")
        save_summary_json(report, out)
        with open(out) as f:
            data = json.load(f)
        assert len(data["runs"]) == 3
