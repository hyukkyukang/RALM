from dataclasses import dataclass
from typing import *

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class State:
    current_input_ids: torch.Tensor
    pad_start_positions: List[int]
    # These are for retrieval-based model
    retrieved_input_ids: Optional[torch.Tensor] = None
    num_retrieval_blocks: Optional[int] = None
    retrieval_block_size: Optional[int] = None
    # This contains all the token ids from all steps (including the first given input token ids)
    all_token_ids: Optional[List[List[int]]] = None
    # Theses are likely to be None for the first step
    position_ids: Optional[torch.LongTensor] = None
    past_key_values: Optional[Union[DynamicCache, List[DynamicCache]]] = None
    # This is for retrieval-based model
    retrieval_key_values: Optional[torch.Tensor] = None
