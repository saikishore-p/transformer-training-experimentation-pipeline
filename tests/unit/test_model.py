"""
Unit tests for the model layer.

Uses a lightweight fake nn.Module for most tests to avoid model downloads.
One smoke test loads distilbert-base-uncased (cached after first run).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from pipeline.config import ModelConfig
from pipeline.models import ModelWrapper, create_model
from pipeline.models.hf_model import HuggingFaceModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(name: str = "distilbert-base-uncased") -> ModelConfig:
    return ModelConfig(type="huggingface", name=name)


class _TinyLinear(nn.Module):
    """Minimal nn.Module used to test wrapper logic without real weights."""

    def __init__(self, num_labels: int = 4):
        super().__init__()
        self.fc = nn.Linear(8, num_labels)
        # mimic HF model interface used by HuggingFaceModelWrapper.load()
        self.config = MagicMock()
        self.config.num_labels = num_labels

    def forward(self, x):
        return self.fc(x)

    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "model.pt")


# ---------------------------------------------------------------------------
# ModelWrapper ABC
# ---------------------------------------------------------------------------


class TestModelWrapperABC:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            ModelWrapper()  # type: ignore

    def test_concrete_must_implement_all_abstract_methods(self):
        class Incomplete(ModelWrapper):
            def get_model(self): ...

            # missing get_tokenizer, get_num_parameters, save, load

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore


# ---------------------------------------------------------------------------
# HuggingFaceModelWrapper — patched (no real model download)
# ---------------------------------------------------------------------------


class TestHuggingFaceModelWrapperPatched:
    """Tests that patch AutoModel / AutoTokenizer — fast, no network."""

    def _make_patched_wrapper(self, num_labels: int = 4):
        """Return a wrapper backed by a TinyLinear, fully patched."""
        fake_model = _TinyLinear(num_labels=num_labels)
        fake_tokenizer = MagicMock()
        fake_tokenizer.save_pretrained = MagicMock()

        config = _make_model_config()
        with (
            patch(
                "pipeline.models.hf_model.AutoModelForSequenceClassification.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "pipeline.models.hf_model.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            return HuggingFaceModelWrapper(config, num_labels), fake_model, fake_tokenizer

    def test_get_model_returns_nn_module(self):
        wrapper, fake_model, _ = self._make_patched_wrapper()
        assert wrapper.get_model() is fake_model

    def test_get_tokenizer_returns_tokenizer(self):
        wrapper, _, fake_tokenizer = self._make_patched_wrapper()
        assert wrapper.get_tokenizer() is fake_tokenizer

    def test_get_num_parameters(self):
        wrapper, fake_model, _ = self._make_patched_wrapper()
        expected = sum(p.numel() for p in fake_model.parameters())
        assert wrapper.get_num_parameters() == expected

    def test_get_trainable_parameters_equals_total_when_all_trainable(self):
        wrapper, _, _ = self._make_patched_wrapper()
        assert wrapper.get_trainable_parameters() == wrapper.get_num_parameters()

    def test_get_trainable_parameters_excludes_frozen(self):
        wrapper, fake_model, _ = self._make_patched_wrapper()
        # Freeze all params
        for p in fake_model.parameters():
            p.requires_grad = False
        assert wrapper.get_trainable_parameters() == 0

    def test_num_labels_property(self):
        wrapper, _, _ = self._make_patched_wrapper(num_labels=3)
        assert wrapper.num_labels == 3

    def test_model_name_property(self):
        wrapper, _, _ = self._make_patched_wrapper()
        assert wrapper.model_name == "distilbert-base-uncased"

    def test_repr_contains_key_info(self):
        wrapper, _, _ = self._make_patched_wrapper(num_labels=4)
        r = repr(wrapper)
        assert "distilbert-base-uncased" in r
        assert "4" in r

    def test_save_calls_save_pretrained(self, tmp_path):
        wrapper, fake_model, fake_tokenizer = self._make_patched_wrapper()
        fake_model.save_pretrained = MagicMock()
        wrapper.save(str(tmp_path / "ckpt"))
        fake_model.save_pretrained.assert_called_once()
        fake_tokenizer.save_pretrained.assert_called_once()

    def test_save_creates_directory(self, tmp_path):
        wrapper, fake_model, fake_tokenizer = self._make_patched_wrapper()
        fake_model.save_pretrained = MagicMock()
        save_path = tmp_path / "deep" / "nested" / "ckpt"
        wrapper.save(str(save_path))
        assert save_path.exists()


# ---------------------------------------------------------------------------
# create_model factory
# ---------------------------------------------------------------------------


class TestCreateModelFactory:
    def test_returns_hf_wrapper_for_huggingface_type(self):
        config = _make_model_config()
        fake_model = _TinyLinear(num_labels=4)
        fake_tokenizer = MagicMock()
        with (
            patch(
                "pipeline.models.hf_model.AutoModelForSequenceClassification.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "pipeline.models.hf_model.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            wrapper = create_model(config, num_labels=4)
        assert isinstance(wrapper, HuggingFaceModelWrapper)

    def test_raises_for_unknown_type(self):
        config = ModelConfig.model_construct(type="custom", name="gpt2")
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(config, num_labels=4)

    def test_wrapper_is_model_wrapper_instance(self):
        config = _make_model_config()
        fake_model = _TinyLinear(num_labels=2)
        fake_tokenizer = MagicMock()
        with (
            patch(
                "pipeline.models.hf_model.AutoModelForSequenceClassification.from_pretrained",
                return_value=fake_model,
            ),
            patch(
                "pipeline.models.hf_model.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ),
        ):
            wrapper = create_model(config, num_labels=2)
        assert isinstance(wrapper, ModelWrapper)


# ---------------------------------------------------------------------------
# Smoke test — real distilbert download (cached after first run, ~260MB)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHuggingFaceModelWrapperReal:
    """
    Uses a real distilbert-base-uncased model.
    Marked `slow` — skipped by default unless -m slow is passed.
    Run with: uv run pytest -m slow tests/unit/test_model.py
    """

    @pytest.fixture(scope="class")
    def wrapper(self):
        config = _make_model_config("distilbert-base-uncased")
        return HuggingFaceModelWrapper(config, num_labels=4)

    def test_model_is_nn_module(self, wrapper):
        assert isinstance(wrapper.get_model(), nn.Module)

    def test_tokenizer_loaded(self, wrapper):
        tok = wrapper.get_tokenizer()
        ids = tok("hello world", return_tensors="pt")
        assert "input_ids" in ids

    def test_num_parameters_nonzero(self, wrapper):
        assert wrapper.get_num_parameters() > 0

    def test_save_and_load_roundtrip(self, wrapper, tmp_path):
        save_path = str(tmp_path / "distilbert_ckpt")
        wrapper.save(save_path)
        config = _make_model_config("distilbert-base-uncased")
        loaded = HuggingFaceModelWrapper.load(save_path, config)
        assert loaded.num_labels == wrapper.num_labels
        assert loaded.get_num_parameters() == wrapper.get_num_parameters()
