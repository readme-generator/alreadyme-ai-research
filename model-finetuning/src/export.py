from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from lightning import MyLightningModule
from modeling import merge_attention_lora_to_single_linear


@torch.inference_mode()
def main(args: argparse.Namespace):
    config = OmegaConf.load(args.config)
    output = config.model.transformer.pretrained_model_name_or_path.split("/")[1]
    output = args.output or output

    model = MyLightningModule.load_from_checkpoint(args.checkpoint, config=config)
    merge_attention_lora_to_single_linear(model)

    model.model.half().save_pretrained(output)
    AutoTokenizer.from_pretrained(**config.model.transformer).save_pretrained(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output")
    main(parser.parse_args())
