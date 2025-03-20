import logging
from typing import *

import torch
from omegaconf import DictConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    Cache,
    FlashAttentionKwargs,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

logger = logging.getLogger("LlamaAttention")


class LlamaAttention(torch.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DictConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = torch.nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = torch.nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        pad_start_positions: Optional[torch.LongTensor] = None,
        **kwargs: FlashAttentionKwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            key_states, value_states = update_dynamic_cache(
                past_key_value,
                key_states,
                value_states,
                self.layer_idx,
                pad_start_positions,
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).repeat(
                1, query_states.shape[1], 1, 1
            )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def update_dynamic_cache(
    dynamic_cache: DynamicCache,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    pad_start_positions: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

    Parameters:
        key_states (`torch.Tensor`):
            The new key states to cache.
        value_states (`torch.Tensor`):
            The new value states to cache.
        layer_idx (`int`):
            The index of the layer to cache the states for.
        cache_kwargs (`Dict[str, Any]`, `optional`):
            Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

    Return:
        A tuple containing the updated key and value states.
    """

    # Update the number of seen tokens
    if layer_idx == 0:
        dynamic_cache._seen_tokens += key_states.shape[-2]

    # Update the cache
    if key_states is not None:
        if len(dynamic_cache.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(dynamic_cache.key_cache), layer_idx):
                dynamic_cache.key_cache.append([])
                dynamic_cache.value_cache.append([])
            dynamic_cache.key_cache.append(key_states)
            dynamic_cache.value_cache.append(value_states)
        elif len(dynamic_cache.key_cache[layer_idx]) == 0:
            # fills previously skipped layers; checking for tensor causes errors
            dynamic_cache.key_cache[layer_idx] = key_states
            dynamic_cache.value_cache[layer_idx] = value_states
        else:
            if pad_start_positions is None:
                dynamic_cache.key_cache[layer_idx] = torch.cat(
                    [dynamic_cache.key_cache[layer_idx], key_states], dim=-2
                )
                dynamic_cache.value_cache[layer_idx] = torch.cat(
                    [dynamic_cache.value_cache[layer_idx], value_states], dim=-2
                )
            else:
                # Insert the new key and value states starting at the non-pad positions
                new_key_cache: List[torch.Tensor] = []
                new_value_cache: List[torch.Tensor] = []
                for batch_idx, pad_start_idx in enumerate(pad_start_positions):
                    new_key_cache.append(
                        torch.cat(
                            [
                                dynamic_cache.key_cache[layer_idx][
                                    batch_idx, :, :pad_start_idx
                                ],
                                key_states[batch_idx, :, :],
                                dynamic_cache.key_cache[layer_idx][
                                    batch_idx, :, pad_start_idx:
                                ],
                            ],
                            dim=-2,
                        )
                    )
                    new_value_cache.append(
                        torch.cat(
                            [
                                dynamic_cache.value_cache[layer_idx][
                                    batch_idx, :, :pad_start_idx
                                ],
                                value_states[batch_idx, :, :],
                                dynamic_cache.value_cache[layer_idx][
                                    batch_idx, :, pad_start_idx:
                                ],
                            ],
                            dim=-2,
                        )
                    )
                dynamic_cache.key_cache[layer_idx] = torch.stack(new_key_cache)
                dynamic_cache.value_cache[layer_idx] = torch.stack(new_value_cache)

    return dynamic_cache.key_cache[layer_idx], dynamic_cache.value_cache[layer_idx]
