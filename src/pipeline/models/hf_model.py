"""
HuggingFace model wrapper.

Wraps AutoModelForSequenceClassification + AutoTokenizer behind the
ModelWrapper interface so the rest of the pipeline stays framework-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from pipeline.config import ModelConfig
from pipeline.models.base import ModelWrapper


class HuggingFaceModelWrapper(ModelWrapper):
    """
    Loads a pretrained HuggingFace sequence-classification model and its
    tokenizer. Replaces the classification head to match num_labels.

    Args:
        config:     ModelConfig with `name` (HF model ID or local path).
        num_labels: Number of output classes inferred from the dataset.
    """

    def __init__(self, config: ModelConfig, num_labels: int) -> None:
        self._config = config
        self._num_labels = num_labels
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.name)
        self._model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            config.name,
            num_labels=num_labels,
            # Suppress the "weights not used" warning for the classification head
            ignore_mismatched_sizes=True,
        )

    # ------------------------------------------------------------------
    # ModelWrapper interface
    # ------------------------------------------------------------------

    def get_model(self) -> nn.Module:
        return self._model

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    def get_trainable_parameters(self) -> int:
        """Number of parameters that will receive gradients during training."""
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save model weights and tokenizer to `path` directory."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, config: ModelConfig) -> HuggingFaceModelWrapper:
        """
        Reload a previously saved wrapper from a checkpoint directory.
        num_labels is inferred from the saved model config.
        """
        saved_model = AutoModelForSequenceClassification.from_pretrained(path)
        num_labels = saved_model.config.num_labels
        wrapper = cls.__new__(cls)
        wrapper._config = config
        wrapper._num_labels = num_labels
        wrapper._tokenizer = AutoTokenizer.from_pretrained(path)
        wrapper._model = saved_model
        return wrapper

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._config.name

    @property
    def num_labels(self) -> int:
        return self._num_labels

    def __repr__(self) -> str:
        return (
            f"HuggingFaceModelWrapper("
            f"name={self._config.name!r}, "
            f"num_labels={self._num_labels}, "
            f"params={self.get_num_parameters():,})"
        )
