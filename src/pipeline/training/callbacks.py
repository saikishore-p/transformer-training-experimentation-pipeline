"""
Training callbacks: EarlyStopping and CheckpointSaver.

Callbacks are called by the Trainer at the end of each epoch.
They are stateful objects — create a fresh instance per training run.
"""
from __future__ import annotations

from pipeline.models.base import ModelWrapper


class EarlyStopping:
    """
    Monitors validation loss and signals when training should stop.

    Stops training after `patience` consecutive epochs with no
    improvement greater than `min_delta`.

    Args:
        patience:  Number of epochs to wait without improvement before stopping.
        min_delta: Minimum decrease in val_loss to count as an improvement.
    """

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss: float = float("inf")
        self._epochs_without_improvement: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """
        Update state with the latest validation loss.

        Returns True if training should stop, False otherwise.
        Also sets self.should_stop for inspection after the call.
        """
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        self.should_stop = self._epochs_without_improvement >= self.patience
        return self.should_stop

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def epochs_without_improvement(self) -> int:
        return self._epochs_without_improvement


class CheckpointSaver:
    """
    Saves the model whenever validation loss improves (best-only strategy).

    Checkpoints are written to:
        {output_dir}/{experiment_name}/best/

    Only one checkpoint is kept on disk at a time — the best so far.

    Args:
        output_dir:       Root directory for all checkpoints (e.g. "checkpoints").
        experiment_name:  Subdirectory name, typically the experiment name from config.
    """

    def __init__(self, output_dir: str, experiment_name: str) -> None:
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self._best_val_loss: float = float("inf")
        self._best_path: str | None = None

    def save(self, model_wrapper: ModelWrapper, epoch: int, val_loss: float) -> str | None:
        """
        Save the model if val_loss improved since last save.

        Returns the checkpoint path if a save occurred, None otherwise.
        """
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            path = f"{self.output_dir}/{self.experiment_name}/best"
            model_wrapper.save(path)
            self._best_path = path
            return path
        return None

    def get_best_checkpoint_path(self) -> str | None:
        """Return the path of the best checkpoint saved so far, or None."""
        return self._best_path

    @property
    def best_val_loss(self) -> float:
        return self._best_val_loss
