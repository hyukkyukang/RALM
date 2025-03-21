from typing import *

import torch
from transformers.cache_utils import DynamicCache

from src.model.wrapper.next_word_prediction.base import NextWordPredictor
from src.model.wrapper.next_word_prediction.state import State


class NextWordPredictorForLlama(NextWordPredictor):
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
        outputs = self.model(
            state.current_input_ids,
            pad_start_positions=state.pad_start_positions,
            past_key_values=state.past_key_values,
            position_ids=state.position_ids,
            use_cache=True,
        )
        return (
            outputs.logits,
            (outputs.past_key_values, None),
        )
