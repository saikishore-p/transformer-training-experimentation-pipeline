"""
Unit tests for the training pipeline.

All tests use lightweight fake components — no model downloads,
no real GPU required. The fake model/dataset produce valid tensor shapes
so the full Trainer loop executes end-to-end in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

from datasets import Dataset
import pytest
import torch
import torch.nn as nn

from pipeline.config import TrainingConfig
from pipeline.training.callbacks import CheckpointSaver, EarlyStopping
from pipeline.training.trainer import Trainer, TrainingResult, _build_optimizer
from pipeline.training.utils import get_device, set_seed

# ---------------------------------------------------------------------------
# Fake components
# ---------------------------------------------------------------------------


@dataclass
class _FakeOutput:
    """Mimics the HuggingFace model output shape expected by Trainer."""

    loss: torch.Tensor
    logits: torch.Tensor


class _FakeHFModel(nn.Module):
    """
    Tiny model that returns a fake loss and logits.
    Loss = mean(input_ids.float()) so it varies across batches
    (important for testing best-epoch tracking).
    """

    def __init__(self, num_labels: int = 4):
        super().__init__()
        self.fc = nn.Linear(1, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        logits = self.fc(input_ids[:, :1].float())  # (B, num_labels)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        else:
            loss = torch.tensor(0.0, requires_grad=True)
        return _FakeOutput(loss=loss, logits=logits)


def _make_fake_wrapper(num_labels: int = 4) -> MagicMock:
    model = _FakeHFModel(num_labels=num_labels)
    wrapper = MagicMock()
    wrapper.get_model.return_value = model
    wrapper.save = MagicMock()
    return wrapper


def _make_fake_dataset(n: int = 32, seq_len: int = 8, num_labels: int = 4) -> Dataset:
    return Dataset.from_dict(
        {
            "input_ids": [[1] * seq_len] * n,
            "attention_mask": [[1] * seq_len] * n,
            "labels": [i % num_labels for i in range(n)],
        }
    ).with_format("torch")


def _make_stub_tracker():
    """No-op tracker with the same interface as MLflowTracker."""
    tracker = MagicMock()
    tracker.log_metrics = MagicMock()
    tracker.log_metric = MagicMock()
    return tracker


def _make_config(**overrides) -> TrainingConfig:
    defaults = dict(
        learning_rate=1e-3,
        batch_size=8,
        epochs=2,
        optimizer="adamw",
        weight_decay=0.01,
        warmup_ratio=0.1,
        seed=42,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# EarlyStopping tests
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_no_stop_while_improving(self):
        es = EarlyStopping(patience=3)
        assert es.step(1.0) is False
        assert es.step(0.9) is False
        assert es.step(0.8) is False

    def test_stops_after_patience_exceeded(self):
        es = EarlyStopping(patience=2)
        es.step(1.0)  # sets best
        es.step(1.1)  # no improvement (1)
        stopped = es.step(1.1)  # no improvement (2) → stop
        assert stopped is True
        assert es.should_stop is True

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=2)
        es.step(1.0)
        es.step(1.1)  # counter = 1
        es.step(0.5)  # improvement → counter resets
        assert es.epochs_without_improvement == 0
        assert es.should_stop is False

    def test_min_delta_respected(self):
        es = EarlyStopping(patience=1, min_delta=0.1)
        es.step(1.0)
        # 0.95 is an improvement but not by min_delta=0.1
        stopped = es.step(0.95)
        assert stopped is True

    def test_best_loss_tracked(self):
        es = EarlyStopping(patience=3)
        es.step(1.0)
        es.step(0.7)
        es.step(0.9)
        assert es.best_loss == pytest.approx(0.7)

    def test_patience_one_stops_immediately(self):
        es = EarlyStopping(patience=1)
        es.step(1.0)  # sets best
        stopped = es.step(1.0)  # no improvement → stop
        assert stopped is True


# ---------------------------------------------------------------------------
# CheckpointSaver tests
# ---------------------------------------------------------------------------


class TestCheckpointSaver:
    def test_saves_on_first_call(self, tmp_path):
        wrapper = _make_fake_wrapper()
        saver = CheckpointSaver(str(tmp_path), "test_exp")
        path = saver.save(wrapper, epoch=1, val_loss=0.5)
        assert path is not None
        wrapper.save.assert_called_once()

    def test_saves_when_loss_improves(self, tmp_path):
        wrapper = _make_fake_wrapper()
        saver = CheckpointSaver(str(tmp_path), "test_exp")
        saver.save(wrapper, epoch=1, val_loss=0.5)
        path = saver.save(wrapper, epoch=2, val_loss=0.3)
        assert path is not None
        assert wrapper.save.call_count == 2

    def test_no_save_when_loss_does_not_improve(self, tmp_path):
        wrapper = _make_fake_wrapper()
        saver = CheckpointSaver(str(tmp_path), "test_exp")
        saver.save(wrapper, epoch=1, val_loss=0.5)
        path = saver.save(wrapper, epoch=2, val_loss=0.6)
        assert path is None
        assert wrapper.save.call_count == 1  # only first save

    def test_best_checkpoint_path_updated(self, tmp_path):
        wrapper = _make_fake_wrapper()
        saver = CheckpointSaver(str(tmp_path), "my_exp")
        assert saver.get_best_checkpoint_path() is None
        saver.save(wrapper, epoch=1, val_loss=0.5)
        assert saver.get_best_checkpoint_path() is not None
        assert "my_exp" in saver.get_best_checkpoint_path()

    def test_best_val_loss_tracked(self, tmp_path):
        wrapper = _make_fake_wrapper()
        saver = CheckpointSaver(str(tmp_path), "exp")
        saver.save(wrapper, epoch=1, val_loss=0.8)
        saver.save(wrapper, epoch=2, val_loss=0.4)
        saver.save(wrapper, epoch=3, val_loss=0.6)
        assert saver.best_val_loss == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Optimizer builder tests
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def _model(self):
        return nn.Linear(4, 4)

    def test_adamw(self):
        opt = _build_optimizer(TrainingConfig(optimizer="adamw"), self._model())
        assert isinstance(opt, torch.optim.AdamW)

    def test_adam(self):
        opt = _build_optimizer(TrainingConfig(optimizer="adam"), self._model())
        assert isinstance(opt, torch.optim.Adam)

    def test_sgd(self):
        opt = _build_optimizer(TrainingConfig(optimizer="sgd"), self._model())
        assert isinstance(opt, torch.optim.SGD)


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


class TestTrainer:
    def _make_trainer(self, config=None, callbacks=None):
        config = config or _make_config()
        wrapper = _make_fake_wrapper()
        train_ds = _make_fake_dataset(n=32)
        val_ds = _make_fake_dataset(n=16)
        tracker = _make_stub_tracker()
        return Trainer(wrapper, train_ds, val_ds, config, tracker, callbacks), tracker, wrapper

    def test_returns_training_result(self):
        trainer, _, _ = self._make_trainer()
        result = trainer.train()
        assert isinstance(result, TrainingResult)

    def test_result_has_expected_epochs(self):
        config = _make_config(epochs=3)
        trainer, _, _ = self._make_trainer(config=config)
        result = trainer.train()
        assert len(result.per_epoch_metrics) == 3

    def test_best_epoch_within_range(self):
        config = _make_config(epochs=3)
        trainer, _, _ = self._make_trainer(config=config)
        result = trainer.train()
        assert 1 <= result.best_epoch <= 3

    def test_val_accuracy_between_0_and_1(self):
        trainer, _, _ = self._make_trainer()
        result = trainer.train()
        assert 0.0 <= result.best_val_accuracy <= 1.0

    def test_tracker_log_metrics_called_per_epoch(self):
        config = _make_config(epochs=2)
        trainer, tracker, _ = self._make_trainer(config=config)
        trainer.train()
        # Called once per epoch + once for summary system metrics
        assert tracker.log_metrics.call_count == 3

    def test_tracker_receives_expected_keys(self):
        trainer, tracker, _ = self._make_trainer()
        trainer.train()
        # Collect all keys logged across all calls
        all_keys = set()
        for call in tracker.log_metrics.call_args_list:
            all_keys.update(call.args[0].keys())
        for key in ("train_loss", "val_loss", "val_accuracy", "epoch_time_seconds"):
            assert key in all_keys

    def test_early_stopping_cuts_epochs_short(self):
        config = _make_config(epochs=5)
        # Large min_delta means the fake model's tiny loss improvements don't
        # count — early stopping fires after patience=1 epoch without "real" improvement
        es = EarlyStopping(patience=1, min_delta=10.0)
        trainer, _, _ = self._make_trainer(config=config, callbacks=[es])
        result = trainer.train()
        # Stops at epoch 2: best set at epoch 1, no improvement at epoch 2 → stop
        assert len(result.per_epoch_metrics) < 5

    def test_checkpoint_saver_invoked(self, tmp_path):
        saver = CheckpointSaver(str(tmp_path), "trainer_test")
        trainer, _, wrapper = self._make_trainer(callbacks=[saver])
        trainer.train()
        # Model should have been saved at least once
        wrapper.save.assert_called()

    def test_total_time_positive(self):
        trainer, _, _ = self._make_trainer()
        result = trainer.train()
        assert result.total_time_seconds > 0

    def test_samples_per_second_positive(self):
        trainer, _, _ = self._make_trainer()
        result = trainer.train()
        assert result.samples_per_second > 0

    def test_force_cpu_env_var(self, monkeypatch):
        monkeypatch.setenv("FORCE_CPU", "1")
        device = get_device()
        assert device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestUtils:
    def test_get_device_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("mps", "cpu", "cuda")

    def test_force_cpu_overrides_mps(self, monkeypatch):
        monkeypatch.setenv("FORCE_CPU", "1")
        assert get_device() == torch.device("cpu")

    def test_set_seed_is_deterministic(self):
        set_seed(0)
        a = torch.rand(4)
        set_seed(0)
        b = torch.rand(4)
        assert torch.allclose(a, b)

    def test_different_seeds_produce_different_values(self):
        set_seed(0)
        a = torch.rand(4)
        set_seed(99)
        b = torch.rand(4)
        assert not torch.allclose(a, b)
