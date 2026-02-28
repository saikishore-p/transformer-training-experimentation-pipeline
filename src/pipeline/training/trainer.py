"""
Core training loop.

Runs forward pass → loss → backprop → optimizer step for every epoch,
logs per-epoch metrics via the injected MLflow tracker, and invokes
callbacks (EarlyStopping, CheckpointSaver) at the end of each epoch.

MPS notes:
  - pin_memory=False  (pinned memory is CUDA-only)
  - All tensors are moved to device inside the loop, not at dataset level
  - FORCE_CPU=1 env-var lets you bypass MPS if an op is unsupported
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import psutil
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from pipeline.config import TrainingConfig
from pipeline.models.base import ModelWrapper
from pipeline.training.callbacks import CheckpointSaver, EarlyStopping
from pipeline.training.utils import get_device, set_seed


@dataclass
class TrainingResult:
    """Captures everything produced by a completed training run."""
    best_val_loss: float
    best_val_accuracy: float
    best_epoch: int                   # 1-indexed epoch where best val was reached
    total_time_seconds: float
    samples_per_second: float         # averaged over all training epochs
    checkpoint_path: str | None       # path to best saved checkpoint, or None
    per_epoch_metrics: list[dict] = field(default_factory=list)


def _build_optimizer(
    config: TrainingConfig,
    model: nn.Module,
) -> torch.optim.Optimizer:
    params = model.parameters()
    name = config.optimizer
    if name == "adamw":
        return AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    if name == "adam":
        return Adam(params, lr=config.learning_rate)
    if name == "sgd":
        return SGD(params, lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'")


def _peak_memory_mb() -> float:
    """Current process RSS in MB (cross-platform; works on MPS and CPU)."""
    return psutil.Process().memory_info().rss / (1024 ** 2)


class Trainer:
    """
    Config-driven training loop for any ModelWrapper backend.

    The tracker parameter uses a duck-typed interface (log_metric / log_metrics)
    so the trainer stays decoupled from MLflow internals.  Pass a no-op stub
    in unit tests.

    Args:
        model_wrapper:  Any ModelWrapper implementation.
        train_dataset:  Tokenized HuggingFace Dataset with torch format.
        val_dataset:    Tokenized HuggingFace Dataset with torch format.
        config:         TrainingConfig (lr, batch_size, epochs, …).
        tracker:        Object with log_metric(key, value, step) interface.
        callbacks:      Optional list of [EarlyStopping, CheckpointSaver].
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        train_dataset,
        val_dataset,
        config: TrainingConfig,
        tracker,
        callbacks: list | None = None,
    ) -> None:
        self._wrapper = model_wrapper
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._config = config
        self._tracker = tracker
        self._callbacks: list = callbacks or []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> TrainingResult:
        set_seed(self._config.seed)
        device = get_device()

        model = self._wrapper.get_model()
        model.to(device)

        train_loader = DataLoader(
            self._train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            pin_memory=False,  # pin_memory requires CUDA
        )
        val_loader = DataLoader(
            self._val_dataset,
            batch_size=self._config.batch_size * 2,  # larger batch is fine for eval
            shuffle=False,
            pin_memory=False,
        )

        optimizer = _build_optimizer(self._config, model)

        total_train_steps = len(train_loader) * self._config.epochs
        warmup_steps = int(total_train_steps * self._config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )

        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        best_epoch = 1
        checkpoint_path: str | None = None
        per_epoch_metrics: list[dict] = []
        total_samples_trained = 0
        run_start = time.perf_counter()

        early_stopper = next(
            (cb for cb in self._callbacks if isinstance(cb, EarlyStopping)), None
        )
        ckpt_saver = next(
            (cb for cb in self._callbacks if isinstance(cb, CheckpointSaver)), None
        )

        for epoch in range(1, self._config.epochs + 1):
            epoch_start = time.perf_counter()

            # ---- training pass ----
            train_loss = self._run_train_epoch(model, train_loader, optimizer, scheduler, device)
            total_samples_trained += len(self._train_dataset)

            # ---- validation pass ----
            val_loss, val_accuracy = self._run_val_epoch(model, val_loader, device)

            epoch_time = time.perf_counter() - epoch_start
            samples_per_sec = len(self._train_dataset) / epoch_time

            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch_time_seconds": epoch_time,
                "samples_per_second": samples_per_sec,
            }
            per_epoch_metrics.append(epoch_metrics)

            # Log to tracker (MLflow or stub)
            self._tracker.log_metrics(epoch_metrics, step=epoch)

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_epoch = epoch

            # Callbacks
            if ckpt_saver is not None:
                saved = ckpt_saver.save(self._wrapper, epoch, val_loss)
                if saved:
                    checkpoint_path = saved

            if early_stopper is not None and early_stopper.step(val_loss):
                break

        total_time = time.perf_counter() - run_start
        avg_samples_per_sec = (
            total_samples_trained / total_time if total_time > 0 else 0.0
        )

        # Log summary system metrics
        self._tracker.log_metrics({
            "total_training_time_seconds": total_time,
            "peak_memory_mb": _peak_memory_mb(),
        })

        return TrainingResult(
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
            best_epoch=best_epoch,
            total_time_seconds=total_time,
            samples_per_second=avg_samples_per_sec,
            checkpoint_path=checkpoint_path,
            per_epoch_metrics=per_epoch_metrics,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
    ) -> float:
        """Run one full training epoch. Returns average batch loss."""
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            # Gradient clipping prevents exploding gradients on large models
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _run_val_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> tuple[float, float]:
        """Run one full validation pass. Returns (avg_loss, accuracy)."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        avg_loss = total_loss / len(loader)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy
