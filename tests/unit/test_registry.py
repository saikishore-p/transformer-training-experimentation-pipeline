"""
Unit tests for ModelRegistry.

All tests use tmp_path so no real checkpoints directory is written
to the project. Tests verify persistence (reload from disk), best-model
tracking correctness, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.registry import ModelRegistry, RegistryEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reg(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(str(tmp_path / "registry.json"))


def _register(registry: ModelRegistry, run_id: str, accuracy: float, **kwargs) -> RegistryEntry:
    defaults = dict(
        experiment_name="test-exp",
        model_name="distilbert-base-uncased",
        checkpoint_path=f"checkpoints/{run_id}/best",
        test_accuracy=accuracy,
        val_accuracy=accuracy - 0.01,
        config_snapshot={"training.learning_rate": "2e-05"},
        test_f1=accuracy - 0.02,
    )
    defaults.update(kwargs)
    return registry.register(run_id=run_id, **defaults)


# ---------------------------------------------------------------------------
# RegistryEntry dataclass
# ---------------------------------------------------------------------------


class TestRegistryEntry:
    def test_to_dict_round_trip(self):
        entry = RegistryEntry(
            run_id="abc",
            experiment_name="exp",
            model_name="distilbert",
            checkpoint_path="ckpt/",
            test_accuracy=0.9,
            val_accuracy=0.89,
            config_snapshot={"lr": "2e-5"},
            registered_at="2026-01-01T00:00:00+00:00",
        )
        d = entry.to_dict()
        restored = RegistryEntry.from_dict(d)
        assert restored.run_id == "abc"
        assert restored.test_accuracy == pytest.approx(0.9)

    def test_from_dict_ignores_unknown_keys(self):
        d = {
            "run_id": "x",
            "experiment_name": "e",
            "model_name": "m",
            "checkpoint_path": "p",
            "test_accuracy": 0.8,
            "val_accuracy": 0.79,
            "config_snapshot": {},
            "registered_at": "2026-01-01",
            "future_unknown_field": "ignored",
        }
        entry = RegistryEntry.from_dict(d)
        assert entry.run_id == "x"

    def test_optional_fields_default_to_zero(self):
        entry = RegistryEntry(
            run_id="r",
            experiment_name="e",
            model_name="m",
            checkpoint_path="p",
            test_accuracy=0.5,
            val_accuracy=0.49,
            config_snapshot={},
            registered_at="2026-01-01",
        )
        assert entry.test_f1 == 0.0
        assert entry.num_parameters == 0
        assert entry.total_training_time_seconds == 0.0


# ---------------------------------------------------------------------------
# Registry creation & persistence
# ---------------------------------------------------------------------------


class TestModelRegistryInit:
    def test_creates_registry_file(self, tmp_path):
        _reg(tmp_path)
        assert (tmp_path / "registry.json").exists()

    def test_empty_registry_has_no_entries(self, tmp_path):
        reg = _reg(tmp_path)
        assert len(reg) == 0

    def test_empty_registry_best_model_is_none(self, tmp_path):
        reg = _reg(tmp_path)
        assert reg.get_best_model() is None

    def test_creates_parent_directory(self, tmp_path):
        deep_path = tmp_path / "deep" / "nested" / "registry.json"
        ModelRegistry(str(deep_path))
        assert deep_path.exists()

    def test_reloads_from_existing_file(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "run1", 0.9)
        # Create a fresh instance pointing to same file
        reg2 = _reg(tmp_path)
        assert len(reg2) == 1
        assert reg2.get_entry("run1") is not None


# ---------------------------------------------------------------------------
# Register & retrieve
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_returns_entry(self, tmp_path):
        reg = _reg(tmp_path)
        entry = _register(reg, "run1", 0.85)
        assert isinstance(entry, RegistryEntry)
        assert entry.run_id == "run1"

    def test_registered_entry_retrievable(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "run1", 0.85)
        assert reg.get_entry("run1") is not None

    def test_len_increments_on_register(self, tmp_path):
        reg = _reg(tmp_path)
        assert len(reg) == 0
        _register(reg, "r1", 0.8)
        assert len(reg) == 1
        _register(reg, "r2", 0.85)
        assert len(reg) == 2

    def test_registered_at_is_set(self, tmp_path):
        reg = _reg(tmp_path)
        entry = _register(reg, "r1", 0.9)
        assert entry.registered_at != ""
        assert "2026" in entry.registered_at or "20" in entry.registered_at

    def test_re_registration_replaces_entry(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.8)
        _register(reg, "r1", 0.9)  # same run_id, better accuracy
        assert len(reg) == 1
        assert reg.get_entry("r1").test_accuracy == pytest.approx(0.9)

    def test_config_snapshot_stored(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.8, config_snapshot={"training.learning_rate": "3e-5"})
        entry = reg.get_entry("r1")
        assert entry.config_snapshot["training.learning_rate"] == "3e-5"

    def test_persists_to_json(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.88)
        with open(tmp_path / "registry.json") as f:
            data = json.load(f)
        assert len(data["entries"]) == 1
        assert data["entries"][0]["run_id"] == "r1"


# ---------------------------------------------------------------------------
# Best model tracking
# ---------------------------------------------------------------------------


class TestGetBestModel:
    def test_first_registered_is_best(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.8)
        assert reg.get_best_model().run_id == "r1"

    def test_higher_accuracy_becomes_best(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.80)
        _register(reg, "r2", 0.92)
        assert reg.get_best_model().run_id == "r2"

    def test_lower_accuracy_does_not_replace_best(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.92)
        _register(reg, "r2", 0.80)
        assert reg.get_best_model().run_id == "r1"

    def test_best_persists_across_reload(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.80)
        _register(reg, "r2", 0.95)
        reg2 = _reg(tmp_path)
        assert reg2.get_best_model().run_id == "r2"

    def test_best_accuracy_value_correct(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.80)
        _register(reg, "r2", 0.92)
        _register(reg, "r3", 0.87)
        assert reg.get_best_model().test_accuracy == pytest.approx(0.92)

    def test_tied_accuracy_keeps_first(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.90)
        _register(reg, "r2", 0.90)  # same accuracy
        # First registered stays best (not strictly better)
        assert reg.get_best_model().run_id == "r1"


# ---------------------------------------------------------------------------
# get_all_entries ordering
# ---------------------------------------------------------------------------


class TestGetAllEntries:
    def test_returns_all(self, tmp_path):
        reg = _reg(tmp_path)
        for i in range(5):
            _register(reg, f"r{i}", 0.7 + i * 0.05)
        assert len(reg.get_all_entries()) == 5

    def test_sorted_best_first(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.75)
        _register(reg, "r2", 0.90)
        _register(reg, "r3", 0.82)
        entries = reg.get_all_entries()
        accuracies = [e.test_accuracy for e in entries]
        assert accuracies == sorted(accuracies, reverse=True)

    def test_empty_returns_empty_list(self, tmp_path):
        reg = _reg(tmp_path)
        assert reg.get_all_entries() == []


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_existing_entry(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.8)
        removed = reg.remove("r1")
        assert removed is True
        assert len(reg) == 0

    def test_remove_nonexistent_returns_false(self, tmp_path):
        reg = _reg(tmp_path)
        assert reg.remove("nonexistent") is False

    def test_remove_best_recomputes_best(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.92)
        _register(reg, "r2", 0.80)
        reg.remove("r1")
        best = reg.get_best_model()
        assert best is not None
        assert best.run_id == "r2"

    def test_remove_only_entry_best_becomes_none(self, tmp_path):
        reg = _reg(tmp_path)
        _register(reg, "r1", 0.8)
        reg.remove("r1")
        assert reg.get_best_model() is None
