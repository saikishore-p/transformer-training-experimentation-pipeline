"""
Unit tests for run_sweep.py.

Tests the pure-Python parts: argument parsing and result file writing.
Does not invoke Ray or actual training.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from scripts.run_sweep import _save_results


class TestSaveResults:
    def test_creates_json_file(self, tmp_path):
        results = [{"run_id": "r1", "test_accuracy": 0.88, "status": "completed"}]
        out = str(tmp_path / "results.json")
        _save_results(results, out)
        assert Path(out).exists()

    def test_file_is_valid_json(self, tmp_path):
        results = [{"run_id": "r1", "test_accuracy": 0.88}]
        out = str(tmp_path / "results.json")
        _save_results(results, out)
        with open(out) as f:
            loaded = json.load(f)
        assert len(loaded) == 1
        assert loaded[0]["run_id"] == "r1"

    def test_multiple_results_all_written(self, tmp_path):
        results = [
            {"run_id": "r1", "test_accuracy": 0.88},
            {"run_id": "r2", "test_accuracy": 0.91},
            {"run_id": "r3", "test_accuracy": 0.85},
        ]
        out = str(tmp_path / "results.json")
        _save_results(results, out)
        with open(out) as f:
            loaded = json.load(f)
        assert len(loaded) == 3

    def test_non_serialisable_values_coerced(self, tmp_path):
        """default=str handles non-JSON types like None, floats."""
        results = [{"status": None, "accuracy": 0.88}]
        out = str(tmp_path / "results.json")
        _save_results(results, out)  # must not raise
        assert Path(out).exists()


class TestRunSweepArgParsing:
    """Test that the CLI rejects bad inputs and accepts valid ones."""

    def _run_main(self, argv: list[str]):
        """Invoke main() with patched sys.argv, returning SystemExit code."""
        with patch("sys.argv", ["run_sweep.py"] + argv):
            import importlib

            from scripts import run_sweep as rs

            importlib.reload(rs)  # fresh parse state
            try:
                rs._parse_args.__wrapped__ if hasattr(rs._parse_args, "__wrapped__") else None
                # Call _parse_args directly
                with patch("sys.argv", ["run_sweep.py"] + argv):
                    return rs._parse_args()
            except SystemExit as e:
                return e

    def test_sweep_arg_required(self):
        with patch("sys.argv", ["run_sweep.py"]):
            from scripts.run_sweep import _parse_args

            with pytest.raises(SystemExit) as exc_info:
                _parse_args()
            assert exc_info.value.code != 0

    def test_valid_sweep_arg_parses(self):
        with patch("sys.argv", ["run_sweep.py", "--sweep", "configs/experiments/lr_sweep.yaml"]):
            from scripts.run_sweep import _parse_args

            args = _parse_args()
            assert args.sweep == "configs/experiments/lr_sweep.yaml"

    def test_max_concurrent_default_is_two(self):
        with patch("sys.argv", ["run_sweep.py", "--sweep", "some.yaml"]):
            from scripts.run_sweep import _parse_args

            args = _parse_args()
            assert args.max_concurrent == 2

    def test_max_concurrent_can_be_set(self):
        with patch("sys.argv", ["run_sweep.py", "--sweep", "some.yaml", "--max-concurrent", "4"]):
            from scripts.run_sweep import _parse_args

            args = _parse_args()
            assert args.max_concurrent == 4

    def test_output_default_is_none(self):
        with patch("sys.argv", ["run_sweep.py", "--sweep", "some.yaml"]):
            from scripts.run_sweep import _parse_args

            args = _parse_args()
            assert args.output is None

    def test_registry_default_is_none(self):
        with patch("sys.argv", ["run_sweep.py", "--sweep", "some.yaml"]):
            from scripts.run_sweep import _parse_args

            args = _parse_args()
            assert args.registry is None


# ---------------------------------------------------------------------------
# main() — mocked run_sweep, real registry + reporting
# ---------------------------------------------------------------------------


class TestMainExitCodes:
    """
    Run main() with a real sweep YAML and real registry,
    but with run_sweep patched to return controlled results instantly.
    """

    def _run(
        self, tmp_path: Path, mock_results: list[dict], extra_argv: list[str] | None = None
    ) -> int:
        import yaml as _yaml

        from pipeline.registry import ModelRegistry

        # Write a minimal sweep YAML
        sweep = tmp_path / "sweep.yaml"
        entries = [
            {
                "model": {"type": "huggingface", "name": "distilbert-base-uncased"},
                "training": {"learning_rate": 2e-5, "batch_size": 8, "epochs": 1, "seed": 42},
                "data": {"dataset_name": "ag_news", "max_length": 32},
                "experiment": {
                    "name": f"exp_{i}",
                    "output_dir": str(tmp_path / "ckpt"),
                    "mlflow_tracking_uri": str(tmp_path / "mlruns"),
                    "mlflow_experiment_name": "test",
                },
            }
            for i in range(len(mock_results))
        ]
        sweep.write_text(_yaml.dump(entries))

        registry_path = str(tmp_path / "registry.json")
        output_path = str(tmp_path / "results.json")

        # Pre-populate registry so report generation works
        reg = ModelRegistry(registry_path)
        for r in mock_results:
            if r.get("status") == "completed":
                reg.register(
                    run_id=r["run_id"],
                    experiment_name=r["experiment_name"],
                    model_name="distilbert-base-uncased",
                    checkpoint_path=f"ckpt/{r['run_id']}",
                    test_accuracy=r["test_accuracy"],
                    val_accuracy=r["test_accuracy"] - 0.01,
                    config_snapshot={"training.learning_rate": "2e-05"},
                )

        argv = [
            "run_sweep.py",
            "--sweep",
            str(sweep),
            "--registry",
            registry_path,
            "--output",
            output_path,
        ]
        if extra_argv:
            argv.extend(extra_argv)

        with (
            patch("sys.argv", argv),
            patch("scripts.run_sweep.run_sweep", return_value=mock_results),
        ):
            from scripts import run_sweep as rs

            try:
                rs.main()
                return 0
            except SystemExit as e:
                return e.code

    def _completed(self, run_id: str, acc: float, regression: bool = False) -> dict:
        return {
            "run_id": run_id,
            "experiment_name": run_id,
            "test_accuracy": acc,
            "test_f1": acc - 0.01,
            "regression_detected": regression,
            "status": "completed",
        }

    def _failed(self, run_id: str) -> dict:
        return {
            "run_id": run_id,
            "experiment_name": run_id,
            "test_accuracy": 0.0,
            "test_f1": 0.0,
            "regression_detected": False,
            "status": "failed",
            "error": "crash",
        }

    def test_exit_zero_all_successful_no_regression(self, tmp_path):
        results = [self._completed("r1", 0.88), self._completed("r2", 0.90)]
        assert self._run(tmp_path, results) == 0

    def test_exit_one_on_any_failed_run(self, tmp_path):
        results = [self._completed("r1", 0.88), self._failed("r2")]
        assert self._run(tmp_path, results) == 1

    def test_exit_two_on_regression_no_failures(self, tmp_path):
        results = [
            self._completed("r1", 0.88, regression=True),
            self._completed("r2", 0.90, regression=False),
        ]
        assert self._run(tmp_path, results) == 2

    def test_exit_one_takes_priority_over_regression(self, tmp_path):
        """Failed run (exit 1) checked before regression (exit 2)."""
        results = [self._failed("r1"), self._completed("r2", 0.88, regression=True)]
        assert self._run(tmp_path, results) == 1

    def test_output_json_written(self, tmp_path):
        results = [self._completed("r1", 0.88)]
        out = str(tmp_path / "out.json")
        self._run(tmp_path, results, extra_argv=["--output", out])
        assert Path(out).exists()

    def test_missing_sweep_file_exits_one(self, tmp_path):
        with (
            patch("sys.argv", ["run_sweep.py", "--sweep", str(tmp_path / "nope.yaml")]),
            patch("scripts.run_sweep.run_sweep", return_value=[]),
        ):
            from scripts import run_sweep as rs

            with pytest.raises(SystemExit) as exc:
                rs.main()
            assert exc.value.code == 1
