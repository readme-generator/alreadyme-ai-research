from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import BloomForCausalLM
from transformers.models.bloom.configuration_bloom import BloomConfig


class FastBloomAttention(nn.Module):
    def __init__(self, config: BloomConfig, layer_number: int):
        super().__init__()
        self.n_head = config.n_head
        self.layer_number = layer_number = max(1, layer_number)
        self.norm_factor = layer_number * (config.hidden_size // config.n_head) ** 0.5

        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        alibi: torch.Tensor,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.unflatten(2, (self.n_head, -1)).transpose(1, 2)

        query, key, value = qkv.split(qkv.size(-1) // 3, dim=3)
        if past_key is not None:
            key = torch.cat((past_key, key), dim=2)
        if past_value is not None:
            value = torch.cat((past_value, value), dim=2)

        attn_scores = torch.matmul(query, key.transpose(2, 3)) / self.norm_factor
        attn_scores = attn_scores + alibi / self.layer_number
        attn_scores = attn_scores * self.layer_number + attention_mask

        attn_probs = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
        context_layer = torch.matmul(attn_probs.type_as(value), value)
        context_layer = context_layer.transpose(1, 2).contiguous().flatten(2, 3)
        return self.dense(context_layer), (key, value)


class FastBloomMLP(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.dense_4h_to_h = nn.Linear(4 * config.hidden_size, config.hidden_size)

    def gelu(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(self.gelu(self.dense_h_to_4h(hidden_states)))


class FastBloomBlock(nn.Module):
    def __init__(self, config: BloomConfig, layer_number: int):
        super().__init__()
        self.residual_ln = config.apply_residual_connection_post_layernorm
        eps = config.layer_norm_epsilon

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=eps)
        self.self_attention = FastBloomAttention(config, layer_number=layer_number)

        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=eps)
        self.mlp = FastBloomMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        alibi: torch.Tensor,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ln = self.input_layernorm(hidden_states)
        attn, present = self.self_attention(
            ln, attention_mask, alibi, past_key, past_value
        )
        attn = attn + (ln if self.residual_ln else hidden_states)

        ln = self.post_attention_layernorm(attn)
        mlp = self.mlp(ln) + (ln if self.residual_ln else attn)
        return mlp, present


class FastBloomModel(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        eps = config.layer_norm_epsilon

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_embeddings_layernorm = nn.LayerNorm(config.hidden_size, eps=eps)

        self.h = nn.ModuleList(FastBloomBlock(config, i) for i in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=eps)

    def build_alibi_tensor(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        positions = attention_mask * (attention_mask.cumsum(1) - 1)
        frequences = 1 + torch.arange(self.n_head, device=hidden_states.device)

        slopes = 2 ** (-8 * frequences / self.n_head)
        alibi = slopes[None, :, None] * positions[:, None, :]
        return alibi.type_as(hidden_states)[:, :, None, :]

    def prepare_causal_mask(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_mask = -10000.0 * (1.0 - attention_mask[:, None, None, :])
        attention_mask = attention_mask.type_as(inputs_embeds)

        if inputs_embeds.size(1) > 1:
            mask = inputs_embeds.new_ones(inputs_embeds.size(1), inputs_embeds.size(1))
            mask = (-10000.0 * mask).triu(1)
            mask = mask[None, None, :, :].expand(inputs_embeds.size(0), 1, -1, -1)
            attention_mask = attention_mask + mask
        return attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        hidden_states = self.word_embeddings_layernorm(self.word_embeddings(input_ids))
        alibi = self.build_alibi_tensor(hidden_states, attention_mask)
        causal_mask = self.prepare_causal_mask(hidden_states, attention_mask)

        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.h):
            past = past_key_values[i] if past_key_values else (None, None)
            hidden_states, present = block(hidden_states, causal_mask, alibi, *past)
            presents.append(present)
        return self.ln_f(hidden_states), presents


class FastBloomForCausalLM(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        self.transformer = FastBloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        outputs = self.transformer(input_ids, past_key_values, attention_mask)
        return self.lm_head(outputs[0]), outputs[1]

    @staticmethod
    def from_pretrained(*args: Any, **kwargs: Any) -> FastBloomForCausalLM:
        original_model = BloomForCausalLM.from_pretrained(*args, **kwargs)
        fused_model = FastBloomForCausalLM(original_model.config)
        fused_model.load_state_dict(original_model.state_dict())
        return fused_model
