from __future__ import annotations

import glob
import os
from typing import Any, Optional

import torch
from bitsandbytes.optim import AdamW8bit
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
from modeling import add_lowrank_adapters, convert_model_to_int8


class MyLightningModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(**config.model)

        convert_model_to_int8(self.model)
        add_lowrank_adapters(self.model, adapter_dim=16)

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
        optimizer = AdamW8bit(self.parameter_groups(), **self.config.optim.optimizer)
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
            tokenizer=AutoTokenizer.from_pretrained(**self.config.model),
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
