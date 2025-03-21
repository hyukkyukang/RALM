from dataclasses import dataclass
from typing import *

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class State:
    current_input_ids: torch.Tensor
    pad_start_positions: torch.LongTensor
    retrieved_input_ids: Optional[torch.Tensor] = None
    num_retrieval_blocks: Optional[int] = None
    past_key_values: Optional[Union[DynamicCache, List[DynamicCache]]] = None
    retrieval_key_values: Optional[torch.Tensor] = None
    position_ids: Optional[torch.LongTensor] = None
    all_token_ids: Optional[List[List[int]]] = None
