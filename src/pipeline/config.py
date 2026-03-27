"""
Config system for the training pipeline.

All experiments are driven by YAML config files validated via Pydantic v2.
Every run is fully reproducible from its config alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
import yaml


class ModelConfig(BaseModel):
    # "custom" support will be added in a later phase (Mode B — custom GPT-2)
    type: Literal["huggingface"] = "huggingface"
    name: str = "distilbert-base-uncased"


class TrainingConfig(BaseModel):
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    # Set to a positive integer to enable early stopping
    early_stopping_patience: int | None = None
    seed: int = 42

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        supported = {"adamw", "adam", "sgd"}
        if v.lower() not in supported:
            raise ValueError(f"Optimizer must be one of {supported}, got '{v}'")
        return v.lower()


class DataConfig(BaseModel):
    dataset_name: str = "ag_news"
    # HuggingFace config name (e.g. "default") — None means no sub-config
    dataset_config: str | None = None
    max_length: int = 128
    # Limit training samples for fast dev iterations; None = use full dataset
    subsample_train: int | None = None
    subsample_val: int | None = None


class ExperimentConfig(BaseModel):
    name: str
    output_dir: str = "checkpoints"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "transformer-pipeline"


class PipelineConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    experiment: ExperimentConfig

    def to_flat_dict(self) -> dict:
        """Flatten nested config into a single dict for MLflow param logging."""
        flat: dict = {}
        for section_name, section in self.model_dump().items():
            if isinstance(section, dict):
                for k, v in section.items():
                    flat[f"{section_name}.{k}"] = v
            else:
                flat[section_name] = section
        return flat


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML config file into a PipelineConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
