from typing import *

import torch
from transformers.cache_utils import DynamicCache

from src.model.wrapper.next_word_prediction.base import NextWordPredictor
from src.model.wrapper.next_word_prediction.state import State


class NextWordPredictorForReLlama(NextWordPredictor):
    def call_model(
        self, state: State
    ) -> Tuple[torch.Tensor, Tuple[DynamicCache, torch.Tensor]]:
        outputs = self.model(
            state.current_input_ids,
            pad_start_positions=state.pad_start_positions,
            retrieved_input_ids=state.retrieved_input_ids,
            num_retrieval_blocks=state.num_retrieval_blocks,
            past_key_values=state.past_key_values,
            retrieval_key_values=state.retrieval_key_values,
            position_ids=state.position_ids,
            use_cache=True,
        )
        return (
            outputs.logits,
            (outputs.past_key_values, outputs.retrieval_key_values),
        )
