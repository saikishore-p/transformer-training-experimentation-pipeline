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

    def __init__(self, config: ModelConfig, num_labels: int) -> None:
        self._config = config
        self._num_labels = num_labels
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.name
        )
        self._model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            config.name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    def get_model(self) -> nn.Module:
        return self._model

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    def get_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, config: ModelConfig) -> "HuggingFaceModelWrapper":
        saved_model = AutoModelForSequenceClassification.from_pretrained(path)
        num_labels = saved_model.config.num_labels
        wrapper = cls.__new__(cls)
        wrapper._config = config
        wrapper._num_labels = num_labels
        wrapper._tokenizer = AutoTokenizer.from_pretrained(path)
        wrapper._model = saved_model
        return wrapper

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
