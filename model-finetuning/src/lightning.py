from __future__ import annotations

import glob
import os
from typing import Any, Optional

import numpy as np
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
from modeling import replace_self_attention_linear_with_lora

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class MyLightningModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(**config.model.transformer)

        # Add LoRA layers and freeze other parameters. Note that we will enable the
        # layernorm for word embeddings because with gradient checkpointing, there is a
        # bug that the gradients are not correctly tracked without preceding gradient
        # requirements.
        self.lora_layers = replace_self_attention_linear_with_lora(
            self.model, **config.model.lora
        )
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora_" in name or "word_embeddings_layernorm" in name

        if config.train.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def training_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        loss = self.model(**batch, use_cache=False).loss
        weight_delta_norm = [layer.weight_delta_norm() for layer in self.lora_layers]
        weight_delta_norm = sum(weight_delta_norm) / len(weight_delta_norm)

        self.log("train/loss", loss)
        self.log("train/weight_delta_norm", weight_delta_norm)
        self.log("step", self.global_step)
        return loss

    def parameter_groups(self) -> list[dict[str, Any]]:
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        do_decay = [p for p in parameters if p.ndim >= 2]
        no_decay = [p for p in parameters if p.ndim < 2]
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
        # Get the textfile list and shuffle with fixed random seed to preserve orders.
        filenames = glob.glob(self.config.data.filenames)
        np.random.RandomState(self.config.data.random_state).shuffle(filenames)

        self.dataset = TextFileDataset(
            filenames=filenames,
            tokenizer=AutoTokenizer.from_pretrained(**self.config.model.transformer),
            max_length=self.config.data.max_length,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            self.config.train.batch_size,
            num_workers=os.cpu_count(),
            collate_fn=DefaultDataCollator(),
            persistent_workers=True,
        )
