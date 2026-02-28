from __future__ import annotations

from dataclasses import asdict, dataclass

from pipeline.registry import ModelRegistry


from rich.console import Console
from rich.text import Text
_console = Console()

DEFAULT_REGRESSION_THRESHOLD: float = 0.02

@dataclass
class RegressionResult:
    regression_detected: bool
    current_accuracy: float
    best_accuracy: float
    best_run_id: str | None
    delta: float
    threshold: float

    @property
    def delta_pct(self) -> float:
        return self.delta * 100

    @property
    def is_first_run(self) -> bool:
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

    def __init__(
        self,
        registry: ModelRegistry,
        threshold: float = DEFAULT_REGRESSION_THRESHOLD,
    ) -> None:
        self._registry = registry
        self._threshold = threshold

    def check(self, current_accuracy: float, run_id: str | None = None) -> RegressionResult:
        best = self._registry.get_best_model()

        if best is not None and run_id and best.run_id == run_id:
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
