"""
Regression Detection System.

After every training run, RegressionDetector compares the new model's
test accuracy against the best known model in the registry.

If the drop exceeds a configurable threshold, a regression is flagged —
printed to the console in yellow/red and logged as MLflow tags.

This directly maps to the Apple JD requirement:
  "Monitor pipeline health and model performance,
   helping identify regressions or data issues"
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.text import Text

from pipeline.registry import ModelRegistry

# Default threshold: flag a regression if accuracy drops by more than 2%
DEFAULT_REGRESSION_THRESHOLD: float = 0.02

_console = Console()


@dataclass
class RegressionResult:
    """
    Outcome of a single regression check.

    All floats are raw fractions (0.0–1.0), not percentages.
    Use `delta_pct` for human-readable display.
    """

    regression_detected: bool
    current_accuracy: float
    best_accuracy: float  # accuracy of the best model in registry
    best_run_id: str | None  # run_id of the best model (None if registry empty)
    delta: float  # current - best  (negative = regression)
    threshold: float  # threshold used for this check

    @property
    def delta_pct(self) -> float:
        """Delta expressed as a percentage (e.g. -2.3 means 2.3% drop)."""
        return self.delta * 100

    @property
    def is_first_run(self) -> bool:
        """True when there is no prior model to compare against."""
        return self.best_run_id is None

    def to_dict(self) -> dict:
        return {
            "regression_detected": self.regression_detected,
            "current_accuracy": self.current_accuracy,
            "best_accuracy": self.best_accuracy,
            "best_run_id": self.best_run_id,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
            "threshold": self.threshold,
        }

    def to_mlflow_tags(self) -> dict[str, str]:
        """Flat string tags suitable for MLflow tracker.set_tags()."""
        return {
            "regression_detected": str(self.regression_detected).lower(),
            "accuracy_delta": f"{self.delta_pct:+.2f}%",
            "best_run_id": self.best_run_id or "none",
        }

    def print_summary(self) -> None:
        """
        Print a colour-coded one-liner to the console.

        ✅ green  — no regression (improvement or within threshold)
        ⚠️ yellow — regression detected
        ℹ️ blue   — first run, no prior model to compare
        """
        if self.is_first_run:
            _console.print(
                Text("ℹ️  First run — no prior model to compare against.", style="bold blue")
            )
            return

        if self.regression_detected:
            msg = (
                f"⚠️  Regression detected: {self.delta_pct:+.1f}% accuracy drop  "
                f"(current={self.current_accuracy:.4f}, "
                f"best={self.best_accuracy:.4f}, "
                f"threshold={self.threshold * 100:.1f}%)"
            )
            _console.print(Text(msg, style="bold yellow"))
        else:
            sign = "+" if self.delta >= 0 else ""
            msg = (
                f"✅ No regression.  Delta: {sign}{self.delta_pct:.1f}%  "
                f"(current={self.current_accuracy:.4f}, "
                f"best={self.best_accuracy:.4f})"
            )
            _console.print(Text(msg, style="bold green"))


class RegressionDetector:
    """
    Compares a new run's accuracy against the best model in the registry.

    Args:
        registry:   ModelRegistry to look up the current best model.
        threshold:  Accuracy drop (absolute, 0–1) that triggers a regression flag.
                    Default: 0.02 (2%).
    """

    def __init__(
        self,
        registry: ModelRegistry,
        threshold: float = DEFAULT_REGRESSION_THRESHOLD,
    ) -> None:
        self._registry = registry
        self._threshold = threshold

    def check(self, current_accuracy: float, run_id: str | None = None) -> RegressionResult:
        """
        Check whether `current_accuracy` represents a regression.

        Logic:
            regression = current_accuracy < best_accuracy - threshold

        If the registry is empty (first run), returns a result with
        regression_detected=False and best_run_id=None.

        Args:
            current_accuracy: Test accuracy of the run just completed (0–1).
            run_id:           MLflow run ID of the current run (excluded from
                              best lookup to avoid self-comparison on re-runs).

        Returns:
            RegressionResult with all comparison details.
        """
        best = self._registry.get_best_model()

        # Exclude the current run from the comparison (handles re-registration)
        if best is not None and run_id and best.run_id == run_id:
            # Find the second-best that isn't this run
            all_entries = self._registry.get_all_entries()
            others = [e for e in all_entries if e.run_id != run_id]
            best = others[0] if others else None

        if best is None:
            return RegressionResult(
                regression_detected=False,
                current_accuracy=current_accuracy,
                best_accuracy=current_accuracy,
                best_run_id=None,
                delta=0.0,
                threshold=self._threshold,
            )

        delta = current_accuracy - best.test_accuracy
        regression_detected = delta < -self._threshold

        return RegressionResult(
            regression_detected=regression_detected,
            current_accuracy=current_accuracy,
            best_accuracy=best.test_accuracy,
            best_run_id=best.run_id,
            delta=delta,
            threshold=self._threshold,
        )
