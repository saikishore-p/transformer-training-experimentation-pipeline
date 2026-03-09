"""
Model layer public API + factory.

Usage:
    from pipeline.models import create_model
    wrapper = create_model(config.model, num_labels=4)
"""

from pipeline.config import ModelConfig
from pipeline.models.base import ModelWrapper
from pipeline.models.hf_model import HuggingFaceModelWrapper


def create_model(config: ModelConfig, num_labels: int) -> ModelWrapper:
    """
    Instantiate the correct ModelWrapper for the given config.

    Currently supports:
      - "huggingface": HuggingFaceModelWrapper
    """
    if config.type == "huggingface":
        return HuggingFaceModelWrapper(config, num_labels)
    raise ValueError(f"Unknown model type '{config.type}'. Supported: ['huggingface']")


__all__ = ["ModelWrapper", "HuggingFaceModelWrapper", "create_model"]
