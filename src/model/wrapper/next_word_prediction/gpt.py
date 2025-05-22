from typing import *

import torch
from transformers.cache_utils import DynamicCache

from src.model.wrapper.next_word_prediction.base import NextWordPredictor
from src.model.wrapper.next_word_prediction.state import State


class NextWordPredictorForGPT(NextWordPredictor):
    def state_initialization(
        self,
        input_ids: torch.Tensor,
        pad_start_positions: List[int],
        **kwargs,
    ) -> State:
        # Extract the non-padding token ids
        all_non_padding_token_ids = self.extract_non_padding_token_ids(
            input_ids=input_ids, pad_start_positions=pad_start_positions
        )

        # Initialize the state
        return State(
            current_input_ids=input_ids,
            pad_start_positions=pad_start_positions,
            all_token_ids=all_non_padding_token_ids,
        )

    def call_model(
        self, state: State
    ) -> Tuple[torch.Tensor, Tuple[DynamicCache, None]]:
        """GPT does not support pad_start_positions.
        We need to call it one-by-one for each instance in the batch.
        """
        bsize = len(state.current_input_ids)
        all_logits: List[torch.Tensor] = []
        all_past_key_values: List[DynamicCache] = []

        for b_idx in range(bsize):
            # Remove the padding tokens
            input_ids_wo_padding = state.current_input_ids[
                b_idx, : state.pad_start_positions[b_idx]
            ]

            # If the past_key_values is None, we need to initialize it
            if state.past_key_values is None:
                past_key_values = None
            else:
                past_key_values = state.past_key_values[b_idx]

            # Call the model
            outputs = self.model(
                input_ids_wo_padding.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
            )
            all_logits.append(outputs.logits.squeeze(0))
            all_past_key_values.append(outputs.past_key_values)

        return all_logits, (all_past_key_values, None)
