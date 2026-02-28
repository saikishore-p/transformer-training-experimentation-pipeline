from __future__ import annotations

import hashlib
import json

from datasets import ClassLabel, DatasetDict, load_dataset

from pipeline.config import DataConfig


_DATASETS_WITHOUT_VALIDATION = {"ag_news"} # AG News has only "train" and "test" splits
_VALIDATION_FRACTION = 0.1


def load_dataset_splits(config: DataConfig, seed: int = 42) -> DatasetDict:

    raw = load_dataset(config.dataset_name, config.dataset_config)

    if "validation" in raw:
        train = raw["train"]
        val = raw["validation"]
        test = raw["test"] if "test" in raw else raw["validation"]
    elif config.dataset_name in _DATASETS_WITHOUT_VALIDATION:
        split = raw["train"].train_test_split(
            test_size=_VALIDATION_FRACTION, seed=seed, stratify_by_column="label"
        )
        train = split["train"]
        val = split["test"]
        test = raw["test"]
    else:
        split = raw["train"].train_test_split(test_size=_VALIDATION_FRACTION, seed=seed)
        train = split["train"]
        val = split["test"]
        test = raw.get("test", split["test"])

    if config.subsample_train is not None:
        n = min(config.subsample_train, len(train))
        train = train.shuffle(seed=seed).select(range(n))
    if config.subsample_val is not None:
        n = min(config.subsample_val, len(val))
        val = val.shuffle(seed=seed).select(range(n))

    return DatasetDict({"train": train, "validation": val, "test": test})


def get_num_labels(splits: DatasetDict) -> int:
    train = splits["train"]
    label_feature = train.features.get("label")
    if isinstance(label_feature, ClassLabel):
        return label_feature.num_classes
    return len(set(train["label"]))


def get_label_names(splits: DatasetDict) -> list[str] | None:
    label_feature = splits["train"].features.get("label")
    if isinstance(label_feature, ClassLabel):
        return label_feature.names
    return None


def get_data_metadata(splits: DatasetDict, config: DataConfig) -> dict:
    train = splits["train"]

    label_counts: dict[int, int] = {}
    for label in train["label"]:
        label_counts[int(label)] = label_counts.get(int(label), 0) + 1

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
