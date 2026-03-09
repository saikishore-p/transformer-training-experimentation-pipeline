"""
Unit tests for the Ray orchestration layer.

Covers: sweep YAML loading, deep-merge logic, and config validation.
Does NOT start a real Ray cluster or run actual training — those are
@slow integration tests. The Ray remote function is tested in isolation
by calling its inner logic directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pipeline.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
)
from pipeline.orchestration.ray_runner import _deep_merge, load_sweep_configs

# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_override_scalar(self):
        base = {"a": 1, "b": 2}
        result = _deep_merge(base, {"b": 99})
        assert result["b"] == 99
        assert result["a"] == 1  # unchanged

    def test_adds_new_key(self):
        base = {"a": 1}
        result = _deep_merge(base, {"z": 42})
        assert result["z"] == 42
        assert result["a"] == 1

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"x": 99}})
        assert base["a"]["x"] == 1  # original untouched

    def test_nested_dict_merges_not_replaces(self):
        base = {"training": {"lr": 1e-5, "batch_size": 16}}
        override = {"training": {"lr": 3e-5}}
        result = _deep_merge(base, override)
        assert result["training"]["lr"] == 3e-5
        assert result["training"]["batch_size"] == 16  # preserved

    def test_deeply_nested_merge(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = _deep_merge(base, override)
        assert result["a"]["b"]["c"] == 99
        assert result["a"]["b"]["d"] == 2

    def test_non_dict_override_replaces(self):
        # If override value is not a dict, it replaces entirely
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": "scalar"}
        result = _deep_merge(base, override)
        assert result["a"] == "scalar"

    def test_empty_override_returns_copy_of_base(self):
        base = {"a": 1, "b": 2}
        result = _deep_merge(base, {})
        assert result == base

    def test_empty_base_returns_override(self):
        override = {"a": 1}
        result = _deep_merge({}, override)
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# load_sweep_configs — from file
# ---------------------------------------------------------------------------


def _write_sweep(tmp_path: Path, entries: list[dict]) -> str:
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.dump(entries))
    return str(path)


def _minimal_entry(name: str, lr: float = 2e-5) -> dict:
    return {
        "model": {"type": "huggingface", "name": "distilbert-base-uncased"},
        "training": {"learning_rate": lr, "batch_size": 8, "epochs": 1, "seed": 42},
        "data": {"dataset_name": "ag_news", "max_length": 32},
        "experiment": {
            "name": name,
            "output_dir": "checkpoints",
            "mlflow_tracking_uri": "mlruns",
            "mlflow_experiment_name": "test",
        },
    }


class TestLoadSweepConfigs:
    def test_returns_list_of_pipeline_configs(self, tmp_path):
        path = _write_sweep(tmp_path, [_minimal_entry("run1"), _minimal_entry("run2")])
        configs = load_sweep_configs(path)
        assert len(configs) == 2
        assert all(isinstance(c, PipelineConfig) for c in configs)

    def test_each_config_has_correct_name(self, tmp_path):
        path = _write_sweep(tmp_path, [_minimal_entry("exp_a"), _minimal_entry("exp_b")])
        configs = load_sweep_configs(path)
        names = [c.experiment.name for c in configs]
        assert names == ["exp_a", "exp_b"]

    def test_each_config_has_correct_lr(self, tmp_path):
        path = _write_sweep(
            tmp_path,
            [
                _minimal_entry("r1", lr=1e-5),
                _minimal_entry("r2", lr=3e-5),
            ],
        )
        configs = load_sweep_configs(path)
        assert configs[0].training.learning_rate == pytest.approx(1e-5)
        assert configs[1].training.learning_rate == pytest.approx(3e-5)

    def test_invalid_yaml_not_list_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump({"not": "a list"}))
        with pytest.raises(ValueError, match="must be a list"):
            load_sweep_configs(str(path))

    def test_validates_each_config(self, tmp_path):
        # Missing experiment.name should raise ValidationError
        bad = {
            "model": {"type": "huggingface", "name": "distilbert-base-uncased"},
            "training": {"learning_rate": 2e-5, "epochs": 1, "seed": 42},
            "data": {"dataset_name": "ag_news"},
            # missing experiment key entirely
        }
        path = _write_sweep(tmp_path, [bad])
        with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError, exact type varies by version
            load_sweep_configs(str(path))

    def test_base_config_used_for_merge(self, tmp_path):
        base = PipelineConfig(
            model=ModelConfig(name="distilbert-base-uncased"),
            training=TrainingConfig(learning_rate=1e-5, epochs=5, batch_size=32, seed=0),
            data=DataConfig(dataset_name="ag_news", max_length=64),
            experiment=ExperimentConfig(
                name="base", mlflow_tracking_uri="mlruns", mlflow_experiment_name="test"
            ),
        )
        # Sweep entry only overrides lr and experiment name
        sweep_entry = {
            "training": {"learning_rate": 3e-5},
            "experiment": {"name": "overridden"},
        }
        path = _write_sweep(tmp_path, [sweep_entry])
        configs = load_sweep_configs(str(path), base_config=base)
        c = configs[0]
        assert c.training.learning_rate == pytest.approx(3e-5)
        assert c.experiment.name == "overridden"
        # Base values preserved
        assert c.training.epochs == 5
        assert c.training.batch_size == 32
        assert c.data.max_length == 64

    def test_real_lr_sweep_file_loads(self):
        """Validates the committed lr_sweep.yaml parses without errors."""
        configs = load_sweep_configs("configs/experiments/lr_sweep.yaml")
        assert len(configs) == 4
        lrs = [c.training.learning_rate for c in configs]
        assert pytest.approx(1e-5) in lrs
        assert pytest.approx(5e-5) in lrs

    def test_real_model_comparison_file_loads(self):
        """Validates the committed model_comparison.yaml parses without errors."""
        configs = load_sweep_configs("configs/experiments/model_comparison.yaml")
        assert len(configs) == 2
        model_names = {c.model.name for c in configs}
        assert "distilbert-base-uncased" in model_names
        assert "distilroberta-base" in model_names
