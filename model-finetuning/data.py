from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class TextFileDataset(Dataset):
    filenames: list[str]
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 2048

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with open(self.filenames[index]) as fp:
            encodings = self.tokenizer(
                fp.read(),
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
        encodings["labels"] = encodings["input_ids"].fill_mask(
            encodings["input_ids"] == self.tokenizer.pad_token_id, -100
        )
        return encodings
