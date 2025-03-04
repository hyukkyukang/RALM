import logging
from functools import cached_property
from typing import *

import torch
import torch._dynamo
from omegaconf import DictConfig
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.models.llama.modeling_llama import (ALL_ATTENTION_FUNCTIONS,
                                                      Cache,
                                                      FlashAttentionKwargs,
                                                      apply_rotary_pos_emb,
                                                      eager_attention_forward)

from src.utils import is_torch_compile_possible

if is_torch_compile_possible():
    flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune")

logger = logging.getLogger("ReLlamaAttention")


class ReLlamaAttention(torch.nn.Module):
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
        self._set_document_id(
            max_input_length=self.config.max_position_embeddings,
            input_chunk_size=self.config.input_chunk_size,
            retrieval_chunk_size=self.config.retrieval_chunk_size,
        )

    @cached_property
    def causal_retrieval_block_mask(self) -> Callable:
        # Configs
        num_of_input_chunks = self.config.max_position_embeddings // self.config.input_chunk_size
        num_of_retrieval_chunks = num_of_input_chunks - 1
        retrieval_chunk_size = self.config.retrieval_chunk_size
        # Define the length of query and key/value, and the start index of input chunks
        Q_LEN = self.config.max_position_embeddings
        KV_LEN = Q_LEN + retrieval_chunk_size * num_of_retrieval_chunks
        INPUT_START_IDX = num_of_retrieval_chunks * retrieval_chunk_size
        
        def causal_retrieval_mask_mod(b, h, q_idx, kv_idx):
            # Basic causal mask
            causal_mask = (q_idx + INPUT_START_IDX) >= kv_idx
            # Remove the input chunk to retrieval chunk connection
            causal_mask = torch.where(
                kv_idx <= INPUT_START_IDX, torch.zeros_like(causal_mask, dtype=torch.bool), causal_mask
            )
            # Additional document mask
            document_mask = (
                self.document_id[q_idx + INPUT_START_IDX]
                == self.document_id[kv_idx]
            )
            # Remove input chunk to input chunk connection (should only handle in causal mask)
            document_mask = torch.where(
                kv_idx >= INPUT_START_IDX, torch.zeros_like(document_mask, dtype=torch.bool), document_mask
            )
            return causal_mask | document_mask
        block_mask = create_block_mask(causal_retrieval_mask_mod, 1, 1, Q_LEN, KV_LEN, device=self.document_id.device)
        
        return block_mask

    def _set_document_id(
        self,
        max_input_length: int,
        input_chunk_size: int,
        retrieval_chunk_size: int,
    ):
        num_chunk_per_batch = max_input_length // input_chunk_size - 1
        input_start_idx = num_chunk_per_batch * retrieval_chunk_size

        # Initialize the document id
        all_input_len = max_input_length + retrieval_chunk_size * num_chunk_per_batch
        self.register_buffer("document_id", torch.zeros(all_input_len, dtype=torch.int32))

        # Remove connection between retrieval chunk and input chunk, by setting the input chunk id to be -1
        self.document_id[input_start_idx:] = -1

        # Set the id for each retrieval chunks (begin from id 1, from the first retrieval chunk)
        for i in range(num_chunk_per_batch):
            block_id = i + 1
            start_idx = retrieval_chunk_size * i
            end_idx = retrieval_chunk_size * (i + 1)
            self.document_id[start_idx:end_idx] = block_id

        # Set the id for each input chunks (begin from id 1, from the second input chunk)
        for i in range(1, num_chunk_per_batch + 1):
            block_id = i
            start_idx = input_start_idx + input_chunk_size * i
            end_idx = input_start_idx + input_chunk_size * (i + 1)
            self.document_id[start_idx:end_idx] = block_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        retrieval_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
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
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        if retrieval_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            retrieval_key_states = retrieval_key_value.key_cache[self.layer_idx]
            retrieval_value_states = retrieval_key_value.value_cache[self.layer_idx]
        else:
            retrieval_key_states = None
            retrieval_value_states = None

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            # Check if dtype is compatible with SDPA

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

        # Conduct flex attention if there is retrieval data
        if retrieval_key_value is None:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                is_causal=True,
                **kwargs,
            )
        else:
            attn_output, attn_weights = (
                flex_attention_for_retrieval_augmented_causal_attention(
                    query_states,
                    key_states,
                    value_states,
                    retrieval_key_states,
                    retrieval_value_states,
                    causal_retrieval_block_mask=self.causal_retrieval_block_mask,
                    scaling=self.scaling,
                    return_lse=output_attentions,
                    **kwargs,
                )
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def flex_attention_for_retrieval_augmented_causal_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    retrieval_key_states: torch.Tensor,
    retrieval_value_states: torch.Tensor,
    causal_retrieval_block_mask: Callable,
    scaling: float,
    return_lse: bool = False,
    **kwargs: FlashAttentionKwargs,
) -> Tuple[torch.Tensor, None]:
    b_size = query_states.size(0)
    _, h_n, s_n, h_dim = retrieval_key_states.shape

    # Reshape retrieval key and retrieval values
    retrieval_key_states = (
        retrieval_key_states.reshape(b_size, -1, h_n, s_n, h_dim)
        .transpose(1, 2)
        .reshape(b_size, h_n, -1, h_dim)
    )
    retrieval_value_states = (
        retrieval_value_states.reshape(b_size, -1, h_n, s_n, h_dim)
        .transpose(1, 2)
        .reshape(b_size, h_n, -1, h_dim)
    )

    # Concatenate query, key and value states
    all_query_chunks = query_states
    all_key_chunks = torch.cat([retrieval_key_states, key_states], dim=2)
    all_value_chunks = torch.cat([retrieval_value_states, value_states], dim=2)

    attn_output = flex_attention(
        all_query_chunks,
        all_key_chunks,
        all_value_chunks.float(),
        block_mask=causal_retrieval_block_mask,
        scale=scaling,
        enable_gqa=True,
        return_lse=return_lse,
    )
    return attn_output, None
