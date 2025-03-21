import functools
import logging
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

from src.model.rellama.mask import (
    FLEX_ATT_TORCH_COMPILE_MIN_BLOCK_SIZE,
    generate_causal_retrieval_mask_mod,
)
from src.model.utils import update_dynamic_cache
from src.utils import is_torch_compile_possible

logger = logging.getLogger("ReLlamaAttention")

if is_torch_compile_possible():
    flex_attention = torch.compile(flex_attention)


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
    def get_causal_retrieval_block_mask(
        self,
        input_length: int,
        retrieval_block_num: int,
        kv_with_retrieval_length: Optional[int] = None,
    ) -> Callable:

        # Create the mask mod
        causal_retrieval_mask_mod = generate_causal_retrieval_mask_mod(
            input_length=input_length,
            retrieval_block_num=retrieval_block_num,
            input_chunk_size=self.config.input_chunk_size,
            retrieval_block_size=self.config.retrieval_block_size,
            device=self.device,
        )

        # Create the block mask using the mask mod
        block_mask = create_block_mask(
            causal_retrieval_mask_mod,
            1,
            1,
            input_length,
            kv_with_retrieval_length,
            device=self.device,
        )

        return block_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        pad_start_positions: Optional[torch.LongTensor] = None,
        retrieval_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: FlashAttentionKwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

        # Update the key and value states with the past key and value states
        if past_key_value is not None:
            key_states, value_states = update_dynamic_cache(
                past_key_value,
                key_states,
                value_states,
                self.layer_idx,
                pad_start_positions,
            )

        # Get the retrieved key and value for this layer
        if retrieval_key_value is None:
            retrieval_key_states = None
            retrieval_value_states = None
        else:
            retrieval_key_states, retrieval_value_states = retrieval_key_value[
                self.layer_idx
            ]

        is_using_kv_cache = query_states.shape[2] != key_states.shape[2]
        if is_using_kv_cache:
            # This is when there are past key and value states
            # Which means we have to select the last retrieval block from the past key and value states
            # Use only the last retrieval block if we are using past key and value states
            # Retrieval blocks are already used and is not needed for the current step
            retrieval_key_states, retrieval_value_states = (
                self.get_last_retrieval_block(
                    retrieval_key_states,
                    retrieval_value_states,
                    pad_start_positions=pad_start_positions,
                )
            )

        # Append retrieval blocks to the key and value states
        if retrieval_key_states is not None:
            key_states = torch.cat([retrieval_key_states, key_states], dim=2)
            value_states = torch.cat([retrieval_value_states, value_states], dim=2)

        if retrieval_key_states is not None and not is_using_kv_cache:
            # Conduct flex attention if we need causal retrieval block attention
            retrieval_block_num = (
                retrieval_key_states.shape[2] // self.config.retrieval_block_size
            )
            attn_output, attn_weights = self.safe_flex_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                input_length=input_length,
                retrieval_block_num=retrieval_block_num,
            )
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

    def safe_flex_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        input_length: int,
        retrieval_block_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check if the input length is greater than FLEX_ATT_TORCH_COMPILE_MIN_BLOCK_SIZE
        need_input_preprocess = (
            is_torch_compile_possible()
            and input_length < FLEX_ATT_TORCH_COMPILE_MIN_BLOCK_SIZE
        )

        original_input_length = input_length
        # Extend the query states to FLEX_ATT_TORCH_COMPILE_MIN_BLOCK_SIZE
        if need_input_preprocess:
            bsize, nhead, seq_len, head_dim = query_states.shape
            length_diff = FLEX_ATT_TORCH_COMPILE_MIN_BLOCK_SIZE - seq_len
            tensor_to_extend = torch.zeros(
                bsize,
                nhead,
                length_diff,
                head_dim,
                device=query_states.device,
                dtype=query_states.dtype,
            )
            query_states = torch.cat([query_states, tensor_to_extend], dim=2)
            input_length = FLEX_ATT_TORCH_COMPILE_MIN_BLOCK_SIZE

        # Get the causal retrieval block mask
        causal_retrieval_block_mask = self.get_causal_retrieval_block_mask(
            input_length=input_length,
            retrieval_block_num=retrieval_block_num,
            kv_with_retrieval_length=key_states.shape[2],
        )

        # Conduct flex attention
        attn_output = flex_attention(
            query_states,
            key_states,
            value_states.float(),
            block_mask=causal_retrieval_block_mask,
            scale=self.scaling,
            enable_gqa=True,
        )

        # Cut-off the extended input length
        if need_input_preprocess:
            attn_output = attn_output[:, :, :original_input_length, :]

        return attn_output, None

    def get_last_retrieval_block(
        self,
        retrieval_key_states: torch.Tensor,
        retrieval_value_states: torch.Tensor,
        pad_start_positions: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selected_retrieval_key_states: List[torch.Tensor] = []
        selected_retrieval_value_states: List[torch.Tensor] = []
        # Find out the last retrieval block index range
        for b_idx, pad_start_position in enumerate(pad_start_positions):
            retrieval_block_idx = pad_start_position // self.config.input_chunk_size - 1

            # Handle the case where there is no retrieval block, we just use the first retrieval block as padding
            block_start_idx = max(
                0, retrieval_block_idx * self.config.retrieval_block_size
            )
            # Handle the case where we actually need a new retrieval block
            # Instead of using the new retrieval block, we use the last retrieval block for efficiency
            block_start_idx = min(
                block_start_idx,
                retrieval_key_states.shape[2] - self.config.retrieval_block_size,
            )
            # Find the end index of the retrieval block
            block_end_idx = block_start_idx + self.config.retrieval_block_size

            # Use only the last retrieval block if we are using past key and value states
            # Other retrieval blocks are already used and is not needed for the current step
            selected_retrieval_key_states.append(
                retrieval_key_states[b_idx, :, block_start_idx:block_end_idx, :]
            )
            selected_retrieval_value_states.append(
                retrieval_value_states[b_idx, :, block_start_idx:block_end_idx, :]
            )

        # Concatenate the selected retrieval key and value states
        selected_retrieval_key_states = torch.stack(selected_retrieval_key_states)
        selected_retrieval_value_states = torch.stack(selected_retrieval_value_states)

        # Return the selected retrieval key and value states
        return selected_retrieval_key_states, selected_retrieval_value_states
