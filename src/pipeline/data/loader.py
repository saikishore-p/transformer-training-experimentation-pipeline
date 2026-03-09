"""
Dataset loading for the training pipeline.

Handles HuggingFace dataset downloads, train/val/test splitting,
and deterministic subsampling for fast experimentation.
"""

from __future__ import annotations

import hashlib
import json

from datasets import ClassLabel, DatasetDict, load_dataset

from pipeline.config import DataConfig

# AG News has only "train" and "test" splits — we carve out validation from train
_DATASETS_WITHOUT_VALIDATION = {"ag_news"}
_VALIDATION_FRACTION = 0.1  # 10% of train becomes validation


def load_dataset_splits(config: DataConfig, seed: int = 42) -> DatasetDict:
    """
    Load a HuggingFace dataset and return a DatasetDict with keys:
      "train", "validation", "test"

    For datasets that lack a validation split (e.g. AG News), 10% of
    training data is held out as validation.

    Subsampling is applied after splitting, using a fixed seed so results
    are fully reproducible from config alone.

    Args:
        config: DataConfig with dataset_name, dataset_config, subsample_* fields.
        seed:   Random seed used for splitting and shuffling.

    Returns:
        DatasetDict with "train", "validation", "test" keys.
    """
    raw = load_dataset(config.dataset_name, config.dataset_config)

    # Build train / validation / test
    if "validation" in raw:
        train = raw["train"]
        val = raw["validation"]
        test = raw["test"] if "test" in raw else raw["validation"]
    elif config.dataset_name in _DATASETS_WITHOUT_VALIDATION:
        # Split train → train + validation
        split = raw["train"].train_test_split(
            test_size=_VALIDATION_FRACTION, seed=seed, stratify_by_column="label"
        )
        train = split["train"]
        val = split["test"]
        test = raw["test"]
    else:
        # Fallback: try to make a validation split from train
        split = raw["train"].train_test_split(test_size=_VALIDATION_FRACTION, seed=seed)
        train = split["train"]
        val = split["test"]
        test = raw.get("test", split["test"])

    # Deterministic subsampling
    if config.subsample_train is not None:
        n = min(config.subsample_train, len(train))
        train = train.shuffle(seed=seed).select(range(n))
    if config.subsample_val is not None:
        n = min(config.subsample_val, len(val))
        val = val.shuffle(seed=seed).select(range(n))

    return DatasetDict({"train": train, "validation": val, "test": test})


def get_num_labels(splits: DatasetDict) -> int:
    """Infer number of classification labels from the dataset's label feature."""
    train = splits["train"]
    label_feature = train.features.get("label")
    if isinstance(label_feature, ClassLabel):
        return label_feature.num_classes
    # Fallback: count unique values
    return len(set(train["label"]))


def get_label_names(splits: DatasetDict) -> list[str] | None:
    """Return human-readable label names if available, else None."""
    label_feature = splits["train"].features.get("label")
    if isinstance(label_feature, ClassLabel):
        return label_feature.names
    return None


def get_data_metadata(splits: DatasetDict, config: DataConfig) -> dict:
    """
    Collect dataset statistics for MLflow param/metric logging.

    Returns a flat dict with:
      - dataset_name, dataset_config
      - num_train, num_val, num_test
      - label_distribution_train: JSON string of {label_id: count}
      - dataset_fingerprint: hash of train split for reproducibility tracking
    """
    train = splits["train"]

    # Per-class counts for the training split
    label_counts: dict[int, int] = {}
    for label in train["label"]:
        label_counts[int(label)] = label_counts.get(int(label), 0) + 1

    # Stable fingerprint: hash of train split's info string
    fingerprint_src = json.dumps(
        {"name": config.dataset_name, "num_rows": len(train), "labels": sorted(label_counts)},
        sort_keys=True,
    )
    fingerprint = hashlib.md5(fingerprint_src.encode()).hexdigest()[:8]

    return {
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config or "default",
        "num_train": len(splits["train"]),
        "num_val": len(splits["validation"]),
        "num_test": len(splits["test"]),
        "label_distribution_train": json.dumps(label_counts),
        "dataset_fingerprint": fingerprint,
    }
