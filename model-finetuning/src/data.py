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
            )
        encodings["labels"] = [
            token_id if token_id != self.tokenizer.pad_token_id else -100
            for token_id in encodings["input_ids"]
        ]
        return encodings
