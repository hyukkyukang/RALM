import abc
import itertools
from typing import *

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from src.evaluation.utils import STOPWORDS_FROM_GPT2
from src.model.llama.causal_modeling import LlamaForCausalLM
from src.model.rellama.causal_modeling import ReLlamaForCausalLM
from src.model.wrapper.next_word_prediction.state import State


class NextWordPredictor(abc.ABC):
    def __init__(
        self,
        model: Union[AutoModelForCausalLM, LlamaForCausalLM, ReLlamaForCausalLM],
        tokenizer: AutoTokenizer,
        steps_to_predict: int = 6,
        topk: int = len(STOPWORDS_FROM_GPT2) // 2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.steps_to_predict = steps_to_predict
        self.topk = topk

    @abc.abstractmethod
    def call_model(
        self, state: State
    ) -> Tuple[torch.Tensor, Tuple[DynamicCache, Optional[torch.Tensor]]]:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def predict(
        self,
        input_ids: torch.Tensor,
        pad_start_positions: torch.LongTensor,
        retrieved_input_ids: Optional[torch.Tensor] = None,
        num_retrieval_blocks: Optional[int] = None,
    ) -> List[str]:
        # Configs
        bsize = len(input_ids)

        # List to append the predictions (no padding)
        all_token_ids: List[List[int]] = []
        for b_idx in range(bsize):
            all_token_ids.append(
                input_ids[b_idx, : pad_start_positions[b_idx]].tolist()
            )

        # Initialize the states
        state = State(
            current_input_ids=input_ids,
            pad_start_positions=pad_start_positions,
            retrieved_input_ids=retrieved_input_ids,
            num_retrieval_blocks=num_retrieval_blocks,
            position_ids=None,
            past_key_values=None,
            all_token_ids=all_token_ids,
        )

        # Predict step by step
        for step_idx in range(self.steps_to_predict):
            # Call the model and get the logits and past key values
            logits, (past_key_values, retrieval_key_values) = self.call_model(state)

            # Get topk candidates
            topk_candidates: List[List[int]] = self.get_topk_candidates(
                logits, state.pad_start_positions
            )

            # Get the non-stopword top-1 candidates
            selected_token_ids: List[List[int]] = self.get_non_stopword_top_candidates(
                topk_candidates
            )

            # Update the state
            state = self.update_state(
                state=state,
                past_key_values=past_key_values,
                retrieval_key_values=retrieval_key_values,
                selected_token_ids=selected_token_ids,
                step_idx=step_idx,
            )

        # Decode the final token ids
        return self.tokenizer.batch_decode(state.all_token_ids)

    def update_state(
        self,
        state: State,
        step_idx: int,
        past_key_values: DynamicCache,
        retrieval_key_values: Optional[torch.Tensor],
        selected_token_ids: List[List[int]],
    ) -> State:
        bsize = len(selected_token_ids)

        # Update the states, position ids, and pad start positions
        state.current_input_ids = torch.tensor(selected_token_ids).to(self.model.device)
        state.past_key_values = past_key_values
        state.position_ids = state.pad_start_positions
        state.retrieval_key_values = retrieval_key_values
        # Increment the pad start positions
        if step_idx > 0:
            state.pad_start_positions = [item + 1 for item in state.pad_start_positions]

        # Append for final output containing text from all steps
        for b_idx in range(bsize):
            state.all_token_ids[b_idx].extend(selected_token_ids[b_idx])

        return state

    def get_topk_candidates(
        self,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        pad_start_positions: List[int],
    ) -> List[List[int]]:
        q_len = len(logits[0])

        # Find the last non-pad token position
        last_non_pad_positions = [item - 1 for item in pad_start_positions]

        # Get the logits of the last non-pad tokens
        if q_len == 1:
            # Handle the case where logits is a tensor
            if isinstance(logits, torch.Tensor):
                last_logits = logits[:, 0, :]
            # Handle the case where logits is a list of tensors
            else:
                last_logits = torch.cat(logits, dim=0)
        else:
            last_logits: List[torch.Tensor] = []
            for b_idx, last_non_pad_position in enumerate(last_non_pad_positions):
                # Handle the case where logits is a tensor
                if isinstance(logits, torch.Tensor):
                    last_logits.append(logits[b_idx, last_non_pad_position, :])
                # Handle the case where logits is a list of tensors
                else:
                    last_logits.append(
                        torch.tensor(logits[b_idx][last_non_pad_position, :])
                    )
            last_logits = torch.stack(last_logits)

        _, line_encoded_candidates = torch.topk(
            last_logits,
            k=self.topk,
            dim=-1,
        )
        return line_encoded_candidates.tolist()

    def get_non_stopword_top_candidates(
        self, topk_candidates: List[List[int]]
    ) -> List[List[int]]:
        # Flatten all candidate token IDs from all batches.
        flat_candidate_ids: List[int] = [
            token_id
            for candidate_list in topk_candidates
            for token_id in candidate_list
        ]

        # Batch decode and strip whitespace from each token.
        decoded_tokens = [
            token.strip() for token in self.tokenizer.batch_decode(flat_candidate_ids)
        ]

        # Compute cumulative indices to split the flat list back into batches.
        sizes: List[int] = [len(candidate_list) for candidate_list in topk_candidates]
        indices: List[int] = [0] + list(itertools.accumulate(sizes))

        # For each batch, select the first candidate whose decoded token is not a stopword.
        selected_token_ids: List[List[int]] = [
            [
                next(
                    token_id
                    for token_id, token in zip(
                        topk_candidates[i], decoded_tokens[indices[i] : indices[i + 1]]
                    )
                    if token not in STOPWORDS_FROM_GPT2
                )
            ]
            for i in range(len(topk_candidates))
        ]

        # Check that there is at least one non-stopword candidate in each batch.
        assert all(
            len(token_ids) > 0 for token_ids in selected_token_ids
        ), "There should be at least one non-stopword candidate in each batch."

        return selected_token_ids
