from pipeline.config import ModelConfig
from pipeline.models.base import ModelWrapper
from pipeline.models.hf_model import HuggingFaceModelWrapper


def create_model(config: ModelConfig, num_labels: int) -> ModelWrapper:
    if config.type == "huggingface":
        return HuggingFaceModelWrapper(config, num_labels)
    raise ValueError(
        f"Unknown model type '{config.type}'. Supported: ['huggingface']"
    )


__all__ = ["ModelWrapper", "HuggingFaceModelWrapper", "create_model"]
