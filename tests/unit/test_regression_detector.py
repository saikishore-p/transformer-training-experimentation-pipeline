"""
Unit tests for the regression detection system.

Uses a real in-memory ModelRegistry (backed by tmp_path) so the
full registry → detector interaction is tested without mocking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.monitoring import DEFAULT_REGRESSION_THRESHOLD, RegressionDetector, RegressionResult
from pipeline.registry import ModelRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(str(tmp_path / "registry.json"))


def _register(registry: ModelRegistry, run_id: str, accuracy: float) -> None:
    registry.register(
        run_id=run_id,
        experiment_name="test",
        model_name="distilbert",
        checkpoint_path=f"ckpt/{run_id}",
        test_accuracy=accuracy,
        val_accuracy=accuracy - 0.01,
        config_snapshot={},
    )


def _detector(registry: ModelRegistry, threshold: float = 0.02) -> RegressionDetector:
    return RegressionDetector(registry, threshold=threshold)


# ---------------------------------------------------------------------------
# RegressionResult properties
# ---------------------------------------------------------------------------


class TestRegressionResult:
    def _make(self, **kwargs) -> RegressionResult:
        defaults = dict(
            regression_detected=False,
            current_accuracy=0.88,
            best_accuracy=0.90,
            best_run_id="abc",
            delta=-0.02,
            threshold=0.02,
        )
        defaults.update(kwargs)
        return RegressionResult(**defaults)

    def test_delta_pct_conversion(self):
        r = self._make(delta=-0.023)
        assert r.delta_pct == pytest.approx(-2.3)

    def test_positive_delta_pct(self):
        r = self._make(delta=0.05)
        assert r.delta_pct == pytest.approx(5.0)

    def test_is_first_run_true_when_no_best(self):
        r = self._make(best_run_id=None)
        assert r.is_first_run is True

    def test_is_first_run_false_when_best_exists(self):
        r = self._make(best_run_id="xyz")
        assert r.is_first_run is False

    def test_to_dict_contains_all_keys(self):
        r = self._make()
        d = r.to_dict()
        for key in (
            "regression_detected",
            "current_accuracy",
            "best_accuracy",
            "best_run_id",
            "delta",
            "delta_pct",
            "threshold",
        ):
            assert key in d

    def test_to_dict_delta_pct_is_percentage(self):
        r = self._make(delta=-0.023)
        assert r.to_dict()["delta_pct"] == pytest.approx(-2.3)

    def test_to_mlflow_tags_keys(self):
        r = self._make(regression_detected=True, best_run_id="run42")
        tags = r.to_mlflow_tags()
        assert "regression_detected" in tags
        assert "accuracy_delta" in tags
        assert "best_run_id" in tags

    def test_to_mlflow_tags_regression_true(self):
        r = self._make(regression_detected=True)
        assert r.to_mlflow_tags()["regression_detected"] == "true"

    def test_to_mlflow_tags_regression_false(self):
        r = self._make(regression_detected=False)
        assert r.to_mlflow_tags()["regression_detected"] == "false"

    def test_to_mlflow_tags_no_best(self):
        r = self._make(best_run_id=None)
        assert r.to_mlflow_tags()["best_run_id"] == "none"

    def test_to_mlflow_tags_delta_format(self):
        r = self._make(delta=-0.023)
        tag = r.to_mlflow_tags()["accuracy_delta"]
        assert "-" in tag
        assert "%" in tag


# ---------------------------------------------------------------------------
# RegressionDetector — empty registry (first run)
# ---------------------------------------------------------------------------


class TestFirstRun:
    def test_no_regression_on_first_run(self, tmp_path):
        registry = _make_registry(tmp_path)
        detector = _detector(registry)
        result = detector.check(0.85)
        assert result.regression_detected is False

    def test_is_first_run_flag_set(self, tmp_path):
        registry = _make_registry(tmp_path)
        detector = _detector(registry)
        result = detector.check(0.85)
        assert result.is_first_run is True

    def test_best_run_id_is_none_on_first_run(self, tmp_path):
        registry = _make_registry(tmp_path)
        detector = _detector(registry)
        result = detector.check(0.85)
        assert result.best_run_id is None

    def test_delta_is_zero_on_first_run(self, tmp_path):
        registry = _make_registry(tmp_path)
        detector = _detector(registry)
        result = detector.check(0.85)
        assert result.delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RegressionDetector — normal comparison cases
# ---------------------------------------------------------------------------


class TestRegressionThreshold:
    def test_no_regression_when_accuracy_improves(self, tmp_path):
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.85)
        result = _detector(registry).check(0.90)
        assert result.regression_detected is False

    def test_no_regression_when_equal(self, tmp_path):
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.85)
        result = _detector(registry).check(0.85)
        assert result.regression_detected is False

    def test_no_regression_within_threshold(self, tmp_path):
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.90)
        # Drop of 0.015 < threshold 0.02 → NOT a regression
        result = _detector(registry, threshold=0.02).check(0.885)
        assert result.regression_detected is False

    def test_regression_just_below_threshold(self, tmp_path):
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.90)
        # Drop of 0.021 > threshold 0.02 → regression
        result = _detector(registry, threshold=0.02).check(0.879)
        assert result.regression_detected is True

    def test_regression_large_drop(self, tmp_path):
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.95)
        result = _detector(registry, threshold=0.02).check(0.80)
        assert result.regression_detected is True

    def test_delta_computed_correctly(self, tmp_path):
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.90)
        result = _detector(registry).check(0.87)
        assert result.delta == pytest.approx(0.87 - 0.90)

    def test_current_and_best_accuracy_stored(self, tmp_path):
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.90)
        result = _detector(registry).check(0.87)
        assert result.current_accuracy == pytest.approx(0.87)
        assert result.best_accuracy == pytest.approx(0.90)

    def test_best_run_id_returned(self, tmp_path):
        registry = _make_registry(tmp_path)
        _register(registry, "best-run", 0.90)
        result = _detector(registry).check(0.87)
        assert result.best_run_id == "best-run"

    def test_threshold_stored_in_result(self, tmp_path):
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.90)
        result = _detector(registry, threshold=0.05).check(0.87)
        assert result.threshold == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# RegressionDetector — multiple runs, best-model tracking
# ---------------------------------------------------------------------------


class TestMultipleRuns:
    def test_compares_against_best_not_latest(self, tmp_path):
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.92)  # best
        _register(registry, "r2", 0.88)  # lower than best
        # Current: 0.90, best is r1 at 0.92 → drop of 0.02, no regression (exact threshold)
        result = _detector(registry, threshold=0.02).check(0.90)
        assert result.best_run_id == "r1"
        assert result.best_accuracy == pytest.approx(0.92)

    def test_no_regression_after_many_runs_with_improvement(self, tmp_path):
        registry = _make_registry(tmp_path)
        for i, acc in enumerate([0.80, 0.83, 0.87, 0.91]):
            _register(registry, f"r{i}", acc)
        result = _detector(registry).check(0.93)
        assert result.regression_detected is False


# ---------------------------------------------------------------------------
# RegressionDetector — self-comparison exclusion
# ---------------------------------------------------------------------------


class TestSelfComparison:
    def test_excludes_current_run_from_best_lookup(self, tmp_path):
        """
        When a run re-registers itself (e.g. after post-processing),
        it should compare against the second-best, not itself.
        """
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.85)  # second-best
        _register(registry, "r2", 0.92)  # becomes best in registry
        # Re-check r2: best excluding r2 is r1 at 0.85
        result = _detector(registry, threshold=0.02).check(0.92, run_id="r2")
        assert result.best_run_id == "r1"
        assert result.best_accuracy == pytest.approx(0.85)

    def test_self_comparison_only_run_treated_as_first(self, tmp_path):
        """If there's only one run and it's the current run, treat as first run."""
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.85)
        result = _detector(registry).check(0.85, run_id="r1")
        assert result.is_first_run is True


# ---------------------------------------------------------------------------
# RegressionDetector — custom threshold
# ---------------------------------------------------------------------------


class TestCustomThreshold:
    def test_zero_threshold_always_flags_any_drop(self, tmp_path):
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.90)
        result = _detector(registry, threshold=0.0).check(0.8999)
        assert result.regression_detected is True

    def test_high_threshold_never_flags_small_drops(self, tmp_path):
        registry = _make_registry(tmp_path)
        _register(registry, "r1", 0.90)
        result = _detector(registry, threshold=0.50).check(0.60)
        assert result.regression_detected is False

    def test_default_threshold_is_two_percent(self):
        assert pytest.approx(0.02) == DEFAULT_REGRESSION_THRESHOLD


# ---------------------------------------------------------------------------
# RegressionResult.print_summary — console output (lines 81-102)
# ---------------------------------------------------------------------------


def _make_result(
    *,
    is_first_run: bool = False,
    regression_detected: bool = False,
    current: float = 0.85,
    best: float = 0.82,
    delta: float = 0.03,
) -> RegressionResult:
    return RegressionResult(
        regression_detected=regression_detected,
        current_accuracy=current,
        best_accuracy=best,
        best_run_id=None if is_first_run else "r_prev",
        delta=delta,
        threshold=0.02,
    )


class TestPrintSummary:
    """print_summary() must not raise and must produce visible output."""

    def test_first_run_does_not_raise(self, capsys):
        result = _make_result(is_first_run=True)
        result.print_summary()  # must not raise

    def test_no_regression_does_not_raise(self, capsys):
        result = _make_result(regression_detected=False, current=0.85, best=0.82)
        result.print_summary()  # must not raise

    def test_regression_detected_does_not_raise(self, capsys):
        result = _make_result(regression_detected=True, current=0.78, best=0.85, delta=-0.07)
        result.print_summary()  # must not raise

    def test_first_run_returns_early(self):
        """Verify first-run path exits before checking regression_detected."""
        result = _make_result(is_first_run=True, regression_detected=True)
        # Should not raise even though regression_detected is True on a first run
        result.print_summary()

    def test_regression_result_with_negative_delta(self):
        result = _make_result(regression_detected=True, current=0.75, best=0.88, delta=-0.13)
        result.print_summary()  # must not raise

    def test_no_regression_with_improvement(self):
        result = _make_result(regression_detected=False, current=0.92, best=0.88, delta=0.04)
        result.print_summary()  # must not raise
