import logging
import functools
from typing import *

import torch
import torch._dynamo
from omegaconf import DictConfig
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    Cache,
    FlashAttentionKwargs,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

# from src.utils import is_torch_compile_possible
# if is_torch_compile_possible():
#     # flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune")
#     flex_attention = torch.compile(flex_attention, dynamic=False)

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

    @property
    def device(self) -> torch.device:
        return next(self.q_proj.parameters()).device

    @functools.lru_cache(maxsize=None)
    def get_attention_interface(self, output_attentions: bool = False) -> Callable:
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            # Check if dtype is compatible with SDPA

            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]
        return attention_interface

    @functools.lru_cache(maxsize=None)
    def get_document_ids(
        self,
        input_length: int,
        input_chunk_size: int,
        retrieval_block_size: int,
        retrieval_block_num: int,
    ) -> torch.Tensor:
        input_start_idx = retrieval_block_num * retrieval_block_size

        # Initialize the document id
        all_input_len = input_length + retrieval_block_size * retrieval_block_num
        document_ids = torch.zeros(all_input_len, dtype=torch.int32, device=self.device)

        # Remove connection between retrieval chunk and input chunk, by setting the input chunk id to be -1
        document_ids[input_start_idx:] = -1

        # Set the id for each retrieval blocks (begin from id 1, from the first retrieval chunk)
        for i in range(retrieval_block_num):
            block_id = i + 1
            start_idx = retrieval_block_size * i
            end_idx = retrieval_block_size * (i + 1)
            document_ids[start_idx:end_idx] = block_id

        # Set the id for each input chunks (begin from id 1, from the second input chunk)
        for i in range(1, retrieval_block_num + 1):
            block_id = i
            start_idx = input_start_idx + input_chunk_size * i
            # Handle case where there are less retrieval blocks
            if i == retrieval_block_num:
                end_idx = all_input_len
            else:
                end_idx = input_start_idx + input_chunk_size * (i + 1)
            document_ids[start_idx:end_idx] = block_id

        return document_ids

    @functools.lru_cache(maxsize=None)
    def get_causal_retrieval_block_mask(
        self,
        input_length: int,
        retrieval_block_num: int,
        kv_with_retrieval_length: Optional[int] = None,
    ) -> Callable:
        # Configs
        Q_LEN = input_length
        KV_LEN = kv_with_retrieval_length
        INPUT_START_IDX = retrieval_block_num * self.config.retrieval_block_size

        document_ids = self.get_document_ids(
            input_length=input_length,
            input_chunk_size=self.config.input_chunk_size,
            retrieval_block_size=self.config.retrieval_block_size,
            retrieval_block_num=retrieval_block_num,
        )

        def causal_retrieval_mask_mod(b, h, q_idx, kv_idx):
            # Basic causal mask
            causal_mask = (q_idx + INPUT_START_IDX) >= kv_idx
            # Remove the input chunk to retrieval chunk connection
            causal_mask = torch.where(
                kv_idx <= INPUT_START_IDX,
                False,
                # torch.zeros_like(causal_mask, dtype=torch.bool),
                causal_mask,
            )
            # Additional document mask
            document_mask = (
                document_ids[q_idx + INPUT_START_IDX] == document_ids[kv_idx]
            )
            # Remove input chunk to input chunk connection (should only handle in causal mask)
            document_mask = torch.where(
                kv_idx >= INPUT_START_IDX,
                False,
                # torch.zeros_like(document_mask, dtype=torch.bool),
                document_mask,
            )
            return causal_mask | document_mask

        block_mask = create_block_mask(
            causal_retrieval_mask_mod,
            1,
            1,
            Q_LEN,
            KV_LEN,
            device=document_ids.device,
        )

        return block_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        retrieval_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs: FlashAttentionKwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        b_size = hidden_states.shape[0]
        input_shape = hidden_states.shape[:-1]
        input_length = hidden_states.shape[1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Get the retrieved key and value for this layer
        if retrieval_key_value is None:
            retrieval_key_states = None
            retrieval_value_states = None
        else:
            retrieval_key_states, retrieval_value_states = retrieval_key_value[
                self.layer_idx
            ]

        # Update the key and value states with the past key and value states
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            # Check if using past key and value states
            is_using_past_state = query_states.shape[2] < key_states.shape[2]
            # Use only the last retrieval block if we are using past key and value states
            # Other retrieval blocks are already used and is not needed for the current step
            if is_using_past_state and retrieval_key_value is not None:
                retrieval_key_states = retrieval_key_states[-1:]
                retrieval_value_states = retrieval_value_states[-1:]

        # Append retrieval blocks to the key and value states
        if retrieval_key_states is not None:
            key_states = torch.cat([retrieval_key_states, key_states], dim=2)
            value_states = torch.cat([retrieval_value_states, value_states], dim=2)

        # Conduct flex attention if there is retrieval data
        if self.use_flex_attention(input_length, retrieval_key_states):
            retrieval_block_num = (
                retrieval_key_states.shape[2] // self.config.retrieval_block_size
            )
            causal_retrieval_block_mask = self.get_causal_retrieval_block_mask(
                input_length=input_length,
                retrieval_block_num=retrieval_block_num,
                kv_with_retrieval_length=key_states.shape[2],
            )
            # Conduct flex attention
            attn_output = custom_flex_attention(
                query_states,
                key_states,
                value_states,
                block_mask=causal_retrieval_block_mask,
                scale=self.scaling,
                enable_gqa=True,
                return_lse=output_attentions,
            )
            attn_weights = None
        else:
            attention_interface: Callable = self.get_attention_interface(
                output_attentions=kwargs.get("output_attentions", False)
            )
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
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def use_flex_attention(
        self,
        input_length: int,
        retrieval_key_states: Union[torch.Tensor, None],
    ) -> bool:
        """
        Use flex attention when there is retrieval data and the input length is greater than the input chunk size.
        If the input length is less than the input chunk size, we don't need specific attention mask
        """
        return (
            retrieval_key_states is not None
            and input_length > self.config.input_chunk_size
        )


def custom_flex_attention(
    query_states,
    key_states,
    value_states,
    block_mask,
    scale,
    enable_gqa,
    return_lse,
):
    return flex_attention(
        query_states,
        key_states,
        value_states.float(),
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
    )
