"""Unit tests for the config loading and validation system."""

from pydantic import ValidationError
import pytest

from pipeline.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
    load_config,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.type == "huggingface"
        assert cfg.name == "distilbert-base-uncased"

    def test_custom_name(self):
        cfg = ModelConfig(name="bert-base-uncased")
        assert cfg.name == "bert-base-uncased"


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate == 2e-5
        assert cfg.batch_size == 16
        assert cfg.epochs == 3
        assert cfg.optimizer == "adamw"
        assert cfg.seed == 42
        assert cfg.early_stopping_patience is None

    def test_invalid_optimizer(self):
        with pytest.raises(ValidationError, match="Optimizer must be one of"):
            TrainingConfig(optimizer="rmsprop")

    def test_optimizer_case_insensitive(self):
        cfg = TrainingConfig(optimizer="AdamW")
        assert cfg.optimizer == "adamw"


class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.dataset_name == "ag_news"
        assert cfg.max_length == 128
        assert cfg.subsample_train is None
        assert cfg.subsample_val is None

    def test_subsampling(self):
        cfg = DataConfig(subsample_train=500, subsample_val=100)
        assert cfg.subsample_train == 500
        assert cfg.subsample_val == 100


class TestPipelineConfig:
    def test_requires_experiment_name(self):
        with pytest.raises(ValidationError):
            PipelineConfig()  # experiment.name is required

    def test_valid_minimal_config(self):
        cfg = PipelineConfig(experiment=ExperimentConfig(name="test-run"))
        assert cfg.experiment.name == "test-run"
        assert cfg.model.name == "distilbert-base-uncased"

    def test_to_flat_dict(self):
        cfg = PipelineConfig(experiment=ExperimentConfig(name="test-run"))
        flat = cfg.to_flat_dict()
        assert flat["model.name"] == "distilbert-base-uncased"
        assert flat["training.learning_rate"] == 2e-5
        assert flat["experiment.name"] == "test-run"
        assert flat["data.dataset_name"] == "ag_news"


class TestLoadConfig:
    def test_load_baseline_config(self, tmp_path):
        yaml_content = """
model:
  type: huggingface
  name: distilbert-base-uncased

training:
  learning_rate: 3e-5
  batch_size: 32
  epochs: 5
  seed: 0

data:
  dataset_name: ag_news
  max_length: 64

experiment:
  name: test-load
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)
        cfg = load_config(config_file)
        assert cfg.training.learning_rate == 3e-5
        assert cfg.training.batch_size == 32
        assert cfg.training.epochs == 5
        assert cfg.experiment.name == "test-load"

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("model:\n  type: unknown_type\nexperiment:\n  name: x\n")
        with pytest.raises(ValidationError):
            load_config(config_file)
