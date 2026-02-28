from __future__ import annotations

from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from datasets import Dataset, DatasetDict

from pipeline.config import DataConfig


_KEEP_COLUMNS = {"input_ids", "attention_mask", "labels"}


def tokenize_splits(
    splits: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: DataConfig,
) -> DatasetDict:

    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )

    tokenized: dict[str, Dataset] = {}
    for split_name, dataset in splits.items():
        ds = dataset.map(
            _tokenize,
            batched=True,
            desc=f"Tokenizing {split_name}",
            remove_columns=[c for c in dataset.column_names if c not in ("label",)],
        )
        if "label" in ds.column_names:
            ds = ds.rename_column("label", "labels")

        cols_to_remove = [c for c in ds.column_names if c not in _KEEP_COLUMNS]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)

        ds.set_format(type="torch", columns=list(_KEEP_COLUMNS & set(ds.column_names)))
        tokenized[split_name] = ds

    return DatasetDict(tokenized)


def get_data_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
