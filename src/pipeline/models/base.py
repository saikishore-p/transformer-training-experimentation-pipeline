from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from transformers import PreTrainedTokenizerBase
from pipeline.config import ModelConfig


class ModelWrapper(ABC):

    @abstractmethod
    def get_model(self) -> nn.Module:
        ...

    @abstractmethod
    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        ...

    @abstractmethod
    def get_num_parameters(self) -> int:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str, config: ModelConfig) -> "ModelWrapper":
        ...
