"""
Tokenization and preprocessing for the training pipeline.

Converts raw text datasets into tokenized tensors ready for the model.
Uses HuggingFace's fast tokenizers with batched processing for speed.
"""

from __future__ import annotations

from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from pipeline.config import DataConfig

# Column names we keep after tokenization — everything else is dropped
_KEEP_COLUMNS = {"input_ids", "attention_mask", "labels"}


def tokenize_splits(
    splits: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: DataConfig,
) -> DatasetDict:
    """
    Apply tokenization to every split in the DatasetDict.

    - Truncates to config.max_length
    - Pads to max_length (static padding — simplifies DataLoader collation)
    - Renames "label" → "labels" (HuggingFace models expect "labels")
    - Removes all columns except input_ids, attention_mask, labels
    - Sets format to PyTorch tensors

    Args:
        splits:    DatasetDict with "train", "validation", "test" keys.
        tokenizer: A HuggingFace tokenizer (fast tokenizer recommended).
        config:    DataConfig supplying max_length.

    Returns:
        DatasetDict with the same keys, tokenized and formatted as torch tensors.
    """

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
        # Rename label → labels
        if "label" in ds.column_names:
            ds = ds.rename_column("label", "labels")

        # Drop any remaining columns that aren't needed
        cols_to_remove = [c for c in ds.column_names if c not in _KEEP_COLUMNS]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)

        ds.set_format(type="torch", columns=list(_KEEP_COLUMNS & set(ds.column_names)))
        tokenized[split_name] = ds

    return DatasetDict(tokenized)


def get_data_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    """
    Returns a DataCollatorWithPadding for use in DataLoader.

    Since we already pad to max_length in tokenize_splits, this collator
    acts as a light wrapper that handles any remaining edge cases.
    Note: pad_to_multiple_of is left unset — no CUDA alignment needed on MPS.
    """
    return DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
