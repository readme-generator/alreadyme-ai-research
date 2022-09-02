from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn


class LoRAAttentionQVLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        lora_dim: int = 4,
        lora_scale: float = 8,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.lora_scale = lora_scale
        self.lora_q_A = nn.Parameter(self.weight.new_empty(in_features, lora_dim))
        self.lora_v_A = nn.Parameter(self.weight.new_empty(in_features, lora_dim))
        self.lora_q_B = nn.Parameter(self.weight.new_empty(lora_dim, out_features // 3))
        self.lora_v_B = nn.Parameter(self.weight.new_empty(lora_dim, out_features // 3))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "lora_q_A"):
            nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_v_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_q_B)
            nn.init.zeros_(self.lora_v_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = super().forward(x)
        lora_q = x @ self.lora_q_A @ self.lora_q_B * self.lora_scale
        lora_v = x @ self.lora_v_A @ self.lora_v_B * self.lora_scale
        return result + torch.cat((lora_q, torch.zeros_like(lora_q), lora_v), dim=2)

    @staticmethod
    def from_linear(linear: nn.Linear, **kwargs: Any) -> LoRAAttentionQVLinear:
        lora_linear = LoRAAttentionQVLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            **kwargs,
        )
        lora_linear.weight, lora_linear.bias = linear.weight, linear.bias
        return lora_linear

    def to_linear(self) -> nn.Linear:
        delta_q = (self.lora_q_A @ self.lora_q_B * self.lora_scale).T
        delta_v = (self.lora_v_A @ self.lora_v_B * self.lora_scale).T
        delta = torch.cat((delta_q, torch.zeros_like(delta_q), delta_v), dim=0)

        linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        linear.weight, linear.bias = self.weight + delta, self.bias
        return linear


def replace_self_attention_linear_with_lora(model: nn.Module, **kwargs: Any):
    for m in model.modules():
        for name, child in m.named_children():
            if name == "query_key_value":
                setattr(m, name, LoRAAttentionQVLinear.from_linear(child, **kwargs))


def disable_all_parameters_except_lora(model: nn.Module):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in parameter
