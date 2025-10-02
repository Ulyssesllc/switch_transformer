"""
dataset.py

Utility functions to load datasets for Switch Transformer training.
Supports:
- Toy dataset (random classification data)
- AG News (text classification)
- SST-2 (sentiment analysis, binary classification)
"""

from torch.utils.data import Dataset, TensorDataset
import torch
from typing import Tuple

try:
    from datasets import load_dataset  # type: ignore
except ImportError:
    load_dataset = None


class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels, texts=None):
        self.encodings = encodings
        self.labels = labels
        self.texts = texts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        if self.texts is not None:
            item["text"] = self.texts[idx]
        return item


def get_toy_dataset(num_samples=1000, seq_len=16, vocab_size=500, num_classes=3):
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return dataset


def _normalize_name(name: str) -> str:
    return name.lower().replace("\\", "/").replace("glue/", "").strip()


def get_hf_dataset(
    name="ag_news", tokenizer=None, max_len=64, split="train"
) -> Tuple[Dataset, int]:
    if load_dataset is None:
        raise ImportError("Please install `datasets` library: pip install datasets")

    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for HuggingFace dataset")

    normalized = _normalize_name(name)

    if normalized == "ag_news":
        hf_name = "ag_news"
        subset = None
        actual_split = split
    elif normalized in {"sst2", "sst-2"}:
        hf_name = "glue"
        subset = "sst2"
        if split == "test":
            # GLUE's test split lacks labels; switch to validation unless explicitly overridden
            actual_split = "validation"
        else:
            actual_split = split
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    load_args = (hf_name,) if subset is None else (hf_name, subset)
    ds = load_dataset(*load_args, split=actual_split)

    if hf_name == "ag_news":
        texts = list(ds["text"])
        labels = list(ds["label"])
        num_classes = 4
    else:  # sst2
        column = "sentence" if "sentence" in ds.column_names else "text"
        texts = list(ds[column])
        labels = list(ds["label"])
        num_classes = 2

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_attention_mask=True,
    )
    dataset = TextClassificationDataset(encodings, labels, texts=texts)
    return dataset, num_classes
