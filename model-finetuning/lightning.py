from __future__ import annotations

import glob
import os
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DefaultDataCollator,
    get_scheduler,
)

from data import TextFileDataset
from modeling import (
    disable_all_parameters_except_lora,
    replace_self_attention_linear_with_lora,
)

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class MyLightningModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(**config.model)

        replace_self_attention_linear_with_lora(self.model, lora_dim=4, lora_scale=8)
        disable_all_parameters_except_lora(self.model, ["word_embeddings_layernorm"])

        if config.train.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def training_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        loss = self.model(**batch, use_cache=False).loss
        self.log("train/loss", loss)
        self.log("step", self.global_step)
        return loss

    def parameter_groups(self) -> list[dict[str, Any]]:
        do_decay = [p for p in self.model.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.model.parameters() if p.ndim < 2]
        return [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = AdamW(self.parameter_groups(), **self.config.optim.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **self.config.optim.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if "ApexMixedPrecisionPlugin" in checkpoint:
            checkpoint.pop("ApexMixedPrecisionPlugin")


class MyLightningDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        self.dataset = TextFileDataset(
            filenames=glob.glob(self.config.data.filenames),
            tokenizer=AutoTokenizer.from_pretrained(
                **self.config.model, truncation_side="left"
            ),
            max_length=self.config.data.max_length,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            self.config.train.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            collate_fn=DefaultDataCollator(),
            persistent_workers=True,
        )
