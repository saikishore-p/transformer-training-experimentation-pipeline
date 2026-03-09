"""Unit tests for the data loading and preprocessing layer."""

from __future__ import annotations

from unittest.mock import patch

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
import pytest
from transformers import AutoTokenizer

from pipeline.config import DataConfig
from pipeline.data.loader import (
    get_data_metadata,
    get_label_names,
    get_num_labels,
    load_dataset_splits,
)
from pipeline.data.preprocessor import tokenize_splits

# ---------------------------------------------------------------------------
# Helpers — build a small fake DatasetDict without any network calls
# ---------------------------------------------------------------------------


def _make_fake_splits(
    n_train: int = 100,
    n_val: int = 20,
    n_test: int = 20,
    num_labels: int = 4,
) -> DatasetDict:
    """Create a minimal fake DatasetDict that mirrors AG News structure."""
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(num_classes=num_labels, names=["W", "S", "B", "T"]),
        }
    )

    def _make(n: int) -> Dataset:
        return Dataset.from_dict(
            {
                "text": [f"sample text {i}" for i in range(n)],
                "label": [i % num_labels for i in range(n)],
            },
            features=features,
        )

    return DatasetDict({"train": _make(n_train), "validation": _make(n_val), "test": _make(n_test)})


# ---------------------------------------------------------------------------
# loader.py tests
# ---------------------------------------------------------------------------


class TestGetNumLabels:
    def test_classlabel_feature(self):
        splits = _make_fake_splits(num_labels=4)
        assert get_num_labels(splits) == 4

    def test_classlabel_feature_two_labels(self):
        features = Features(
            {"text": Value("string"), "label": ClassLabel(num_classes=2, names=["ham", "spam"])}
        )
        ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]}, features=features)
        splits = DatasetDict({"train": ds, "validation": ds, "test": ds})
        assert get_num_labels(splits) == 2

    def test_fallback_unique_count(self):
        """Without ClassLabel feature, falls back to counting unique values."""
        ds = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2]})
        splits = DatasetDict({"train": ds, "validation": ds, "test": ds})
        assert get_num_labels(splits) == 3


class TestGetLabelNames:
    def test_returns_names_for_classlabel(self):
        splits = _make_fake_splits(num_labels=4)
        names = get_label_names(splits)
        assert names == ["W", "S", "B", "T"]

    def test_returns_none_without_classlabel(self):
        ds = Dataset.from_dict({"text": ["a"], "label": [0]})
        splits = DatasetDict({"train": ds, "validation": ds, "test": ds})
        assert get_label_names(splits) is None


class TestGetDataMetadata:
    def test_metadata_keys(self):
        splits = _make_fake_splits(n_train=100, n_val=20, n_test=20)
        config = DataConfig(dataset_name="ag_news")
        meta = get_data_metadata(splits, config)
        assert meta["dataset_name"] == "ag_news"
        assert meta["num_train"] == 100
        assert meta["num_val"] == 20
        assert meta["num_test"] == 20
        assert "label_distribution_train" in meta
        assert "dataset_fingerprint" in meta

    def test_label_distribution_is_json_string(self):
        import json

        splits = _make_fake_splits(n_train=40, n_val=10, n_test=10, num_labels=4)
        config = DataConfig(dataset_name="ag_news")
        meta = get_data_metadata(splits, config)
        dist = json.loads(meta["label_distribution_train"])
        # Each label 0-3 should have 10 samples in a 40-sample train set
        assert sum(dist.values()) == 40

    def test_fingerprint_is_short_string(self):
        splits = _make_fake_splits()
        config = DataConfig()
        meta = get_data_metadata(splits, config)
        assert isinstance(meta["dataset_fingerprint"], str)
        assert len(meta["dataset_fingerprint"]) == 8

    def test_dataset_config_default(self):
        splits = _make_fake_splits()
        config = DataConfig(dataset_name="test_ds", dataset_config=None)
        meta = get_data_metadata(splits, config)
        assert meta["dataset_config"] == "default"


# ---------------------------------------------------------------------------
# preprocessor.py tests — use a real (cached) tokenizer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


class TestTokenizeSplits:
    def test_output_columns(self, tokenizer):
        splits = _make_fake_splits(n_train=10, n_val=4, n_test=4)
        config = DataConfig(max_length=32)
        tokenized = tokenize_splits(splits, tokenizer, config)
        for split_name in ("train", "validation", "test"):
            cols = set(tokenized[split_name].column_names)
            assert "input_ids" in cols
            assert "attention_mask" in cols
            assert "labels" in cols
            # Original "text" column must be removed
            assert "text" not in cols
            # "label" should be renamed to "labels", not duplicated
            assert "label" not in cols

    def test_sequence_length_is_capped(self, tokenizer):
        splits = _make_fake_splits(n_train=10, n_val=4, n_test=4)
        config = DataConfig(max_length=16)
        tokenized = tokenize_splits(splits, tokenizer, config)
        for row in tokenized["train"]:
            assert len(row["input_ids"]) == 16

    def test_labels_are_correct(self, tokenizer):
        splits = _make_fake_splits(n_train=8, n_val=4, n_test=4, num_labels=4)
        config = DataConfig(max_length=32)
        tokenized = tokenize_splits(splits, tokenizer, config)
        # Labels should be the original integer labels (0-3 cycling)
        for i, row in enumerate(tokenized["train"]):
            assert int(row["labels"]) == i % 4

    def test_torch_format(self, tokenizer):
        import torch

        splits = _make_fake_splits(n_train=4, n_val=2, n_test=2)
        config = DataConfig(max_length=16)
        tokenized = tokenize_splits(splits, tokenizer, config)
        row = tokenized["train"][0]
        assert isinstance(row["input_ids"], torch.Tensor)
        assert isinstance(row["attention_mask"], torch.Tensor)
        assert isinstance(row["labels"], torch.Tensor)

    def test_subsampling_respected(self, tokenizer):
        """Subsampling happens in loader; preprocessor works on whatever it receives."""
        splits = _make_fake_splits(n_train=20, n_val=5, n_test=5)
        config = DataConfig(max_length=16, subsample_train=10)
        # Manually subsample to simulate loader behaviour
        splits["train"] = splits["train"].select(range(10))
        tokenized = tokenize_splits(splits, tokenizer, config)
        assert len(tokenized["train"]) == 10


# ---------------------------------------------------------------------------
# load_dataset_splits — split-strategy branches (lines 40-70 in loader.py)
# ---------------------------------------------------------------------------


class TestLoadDatasetSplitsWithValidation:
    """Dataset already has a 'validation' key → use it directly."""

    def test_uses_existing_validation_split(self):
        fake = _make_fake_splits(n_train=80, n_val=20, n_test=20)
        with patch("pipeline.data.loader.load_dataset", return_value=fake):
            config = DataConfig(dataset_name="some_dataset_with_val")
            result = load_dataset_splits(config, seed=42)
        assert len(result["validation"]) == 20

    def test_uses_existing_test_split(self):
        fake = _make_fake_splits(n_train=80, n_val=20, n_test=20)
        with patch("pipeline.data.loader.load_dataset", return_value=fake):
            config = DataConfig(dataset_name="some_dataset_with_val")
            result = load_dataset_splits(config, seed=42)
        assert len(result["test"]) == 20

    def test_returns_correct_keys(self):
        fake = _make_fake_splits()
        with patch("pipeline.data.loader.load_dataset", return_value=fake):
            config = DataConfig(dataset_name="some_dataset_with_val")
            result = load_dataset_splits(config, seed=42)
        assert set(result.keys()) == {"train", "validation", "test"}

    def test_subsampling_applied_to_train(self):
        fake = _make_fake_splits(n_train=100, n_val=20, n_test=20)
        with patch("pipeline.data.loader.load_dataset", return_value=fake):
            config = DataConfig(dataset_name="some_dataset_with_val", subsample_train=30)
            result = load_dataset_splits(config, seed=42)
        assert len(result["train"]) == 30

    def test_subsampling_applied_to_val(self):
        fake = _make_fake_splits(n_train=100, n_val=20, n_test=20)
        with patch("pipeline.data.loader.load_dataset", return_value=fake):
            config = DataConfig(dataset_name="some_dataset_with_val", subsample_val=5)
            result = load_dataset_splits(config, seed=42)
        assert len(result["validation"]) == 5

    def test_no_subsampling_keeps_full_size(self):
        fake = _make_fake_splits(n_train=100, n_val=20, n_test=20)
        with patch("pipeline.data.loader.load_dataset", return_value=fake):
            config = DataConfig(dataset_name="some_dataset_with_val")
            result = load_dataset_splits(config, seed=42)
        assert len(result["train"]) == 100


class TestLoadDatasetSplitsWithoutValidation:
    """ag_news path — no validation split, must be carved from train."""

    def _make_ag_news_raw(self, n_train: int = 200, n_test: int = 40) -> DatasetDict:
        """Fake raw DatasetDict with only train + test (like ag_news)."""
        features = Features(
            {
                "text": Value("string"),
                "label": ClassLabel(num_classes=4, names=["W", "S", "B", "T"]),
            }
        )

        def _split(n: int) -> Dataset:
            return Dataset.from_dict(
                {"text": [f"text {i}" for i in range(n)], "label": [i % 4 for i in range(n)]},
                features=features,
            )

        return DatasetDict({"train": _split(n_train), "test": _split(n_test)})

    def test_ag_news_produces_three_splits(self):
        raw = self._make_ag_news_raw()
        with patch("pipeline.data.loader.load_dataset", return_value=raw):
            config = DataConfig(dataset_name="ag_news")
            result = load_dataset_splits(config, seed=42)
        assert set(result.keys()) == {"train", "validation", "test"}

    def test_ag_news_validation_is_carved_from_train(self):
        raw = self._make_ag_news_raw(n_train=200)
        with patch("pipeline.data.loader.load_dataset", return_value=raw):
            config = DataConfig(dataset_name="ag_news")
            result = load_dataset_splits(config, seed=42)
        # ~10% validation, ~90% train
        assert len(result["validation"]) > 0
        assert len(result["train"]) < 200

    def test_ag_news_test_split_preserved(self):
        raw = self._make_ag_news_raw(n_train=200, n_test=40)
        with patch("pipeline.data.loader.load_dataset", return_value=raw):
            config = DataConfig(dataset_name="ag_news")
            result = load_dataset_splits(config, seed=42)
        assert len(result["test"]) == 40

    def test_ag_news_train_plus_val_equals_original_train(self):
        """After splitting, total rows should equal original train size."""
        raw = self._make_ag_news_raw(n_train=200)
        with patch("pipeline.data.loader.load_dataset", return_value=raw):
            config = DataConfig(dataset_name="ag_news")
            result = load_dataset_splits(config, seed=42)
        assert len(result["train"]) + len(result["validation"]) == 200


class TestLoadDatasetSplitsFallback:
    """Unknown dataset with no validation key → generic fallback path."""

    def _make_raw_no_val(self, n_train: int = 200, n_test: int = 40) -> DatasetDict:
        features = Features(
            {
                "text": Value("string"),
                "label": ClassLabel(num_classes=2, names=["neg", "pos"]),
            }
        )

        def _split(n: int) -> Dataset:
            return Dataset.from_dict(
                {"text": [f"t {i}" for i in range(n)], "label": [i % 2 for i in range(n)]},
                features=features,
            )

        return DatasetDict({"train": _split(n_train), "test": _split(n_test)})

    def test_fallback_creates_validation_split(self):
        raw = self._make_raw_no_val()
        with patch("pipeline.data.loader.load_dataset", return_value=raw):
            config = DataConfig(dataset_name="unknown_custom_dataset")
            result = load_dataset_splits(config, seed=42)
        assert "validation" in result
        assert len(result["validation"]) > 0

    def test_fallback_includes_test_from_raw(self):
        raw = self._make_raw_no_val(n_test=40)
        with patch("pipeline.data.loader.load_dataset", return_value=raw):
            config = DataConfig(dataset_name="unknown_custom_dataset")
            result = load_dataset_splits(config, seed=42)
        assert len(result["test"]) == 40

    def test_fallback_train_is_smaller_than_original(self):
        raw = self._make_raw_no_val(n_train=200)
        with patch("pipeline.data.loader.load_dataset", return_value=raw):
            config = DataConfig(dataset_name="unknown_custom_dataset")
            result = load_dataset_splits(config, seed=42)
        assert len(result["train"]) < 200
