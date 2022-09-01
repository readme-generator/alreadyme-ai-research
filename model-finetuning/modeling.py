from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise
from torch import nn


def quantize_blockise_lowmemory(
    matrix: torch.Tensor, chunk_size: int = 2**20
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    code, chunks, absmaxes = None, [], []
    flat_tensor = matrix.flatten()

    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size : (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)

    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    return matrix_i8, (torch.cat(absmaxes), code)


class DequantizeAndLinear(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        input: torch.Tensor,
        weights_quantized: torch.ByteTensor,
        absmax: torch.FloatTensor,
        code: torch.FloatTensor,
        bias: torch.FloatTensor,
    ) -> torch.Tensor:
        weights_deq = dequantize_blockwise(weights_quantized, (absmax, code))
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        input, weights_quantized, absmax, code = ctx.saved_tensors
        weights_deq = dequantize_blockwise(weights_quantized, (absmax, code))

        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias


class FrozenBNBLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        absmax: torch.Tensor,
        code: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.bias = bias
        self.adapter = None

        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))

    def forward(self, x: torch.Tensor):
        weight_args = (self.weight, self.absmax, self.code, self.bias)
        output = DequantizeAndLinear.apply(x, *weight_args)

        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> FrozenBNBLinear:
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)


class FrozenBNBEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.adapter = None

        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))

    def forward(self, input, **kwargs):
        with torch.no_grad():
            weight_deq = dequantize_blockwise(self.weight, (self.absmax, self.code))
            output = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> FrozenBNBEmbedding:
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)


def convert_model_to_int8(model: nn.Module):
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, FrozenBNBLinear.from_linear(child))
            elif isinstance(child, nn.Embedding):
                setattr(module, name, FrozenBNBEmbedding.from_embedding(child))


def add_lowrank_adapters(model: nn.Module, adapter_dim: int = 16):
    for module in model.modules():
        if isinstance(module, FrozenBNBLinear):
            module.adapter = nn.Sequential(
                nn.Linear(module.in_features, adapter_dim, bias=False),
                nn.Linear(adapter_dim, module.out_features, bias=False),
            )
            nn.init.zeros_(module.adapter[1].weight)
        elif isinstance(module, FrozenBNBEmbedding):
            module.adapter = nn.Sequential(
                nn.Embedding(module.num_embeddings, adapter_dim),
                nn.Linear(adapter_dim, module.embedding_dim, bias=False),
            )
            nn.init.zeros_(module.adapter[1].weight)
