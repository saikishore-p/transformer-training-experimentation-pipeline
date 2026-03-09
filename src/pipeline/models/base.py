"""
Abstract base class for all model wrappers.

Defines the interface that both HuggingFace and custom model backends
must implement, making them interchangeable throughout the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from pipeline.config import ModelConfig


class ModelWrapper(ABC):
    """
    Uniform interface over any model backend (HuggingFace, custom GPT-2, etc.).

    The training loop, evaluator, and registry only interact with this
    interface — they never import HuggingFace or custom model classes directly.
    """

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the underlying PyTorch nn.Module."""
        ...

    @abstractmethod
    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the tokenizer associated with this model."""
        ...

    @abstractmethod
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model + tokenizer to the given directory path."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str, config: ModelConfig) -> ModelWrapper:
        """Load a previously saved model + tokenizer from a directory path."""
        ...
