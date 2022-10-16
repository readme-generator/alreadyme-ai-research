from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoConfig, AutoTokenizer

from modeling_fast_bloom import FastBloomForCausalLM


@torch.inference_mode()
def main(args: argparse.Namespace):
    output = args.output or f"compiled-{os.path.basename(args.model)}"
    AutoConfig.from_pretrained(args.model).save_pretrained(output)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(output)

    model = FastBloomForCausalLM.from_pretrained(args.model).eval().cuda().half()
    torch.jit.save(torch.jit.script(model), os.path.join(output, "model.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", nargs="?")
    main(parser.parse_args())
