import itertools
from typing import *

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.utils import STOPWORDS_FROM_GPT2
from src.model.llama.causal_modeling import LlamaForCausalLM
from src.model.rellama.causal_modeling import ReLlamaForCausalLM


class NextWordPredictor:
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

    def predict(
        self,
        input_ids: torch.Tensor,
        pad_start_positions: Optional[torch.LongTensor] = None,
        retrieved_input_ids: Optional[torch.Tensor] = None,
        num_retrieval_blocks: Optional[int] = None,
    ) -> List[str]:
        # Configs
        bsize = len(input_ids)

        # List to append the predictions
        # Remove padding token ids from the input ids
        final_token_ids: List[List[int]] = []
        for b_idx in range(bsize):
            final_token_ids.append(
                input_ids[b_idx, : pad_start_positions[b_idx]].tolist()
            )

        # Predict step by step
        current_input_token_ids = input_ids
        states = None
        position_ids = None
        for step_idx in range(self.steps_to_predict):
            outputs = self.model(
                current_input_token_ids,
                pad_start_positions=pad_start_positions,
                retrieved_input_ids=retrieved_input_ids,
                num_retrieval_blocks=num_retrieval_blocks,
                use_cache=True,
                past_key_values=states,
                position_ids=position_ids,
            )

            # Get topk candidates
            logits = outputs.logits
            topk_candidates: List[List[int]] = self.get_topk_candidates(
                logits, pad_start_positions
            )
            selected_token_ids: List[List[int]] = self.get_non_stopword_top_candidates(
                topk_candidates
            )

            # Update the states, position ids, and pad start positions
            states = outputs.past_key_values
            position_ids = pad_start_positions
            if step_idx > 0:
                pad_start_positions = [item + 1 for item in pad_start_positions]

            # Update the current input token ids
            current_input_token_ids = torch.tensor(selected_token_ids).to(
                self.model.device
            )

            # Append for final output containing text from all steps
            for b_idx in range(bsize):
                final_token_ids[b_idx].extend(selected_token_ids[b_idx])

        # Decode the final token ids
        return self.tokenizer.batch_decode(final_token_ids)

    def get_topk_candidates(
        self, logits: torch.Tensor, pad_start_positions: List[int]
    ) -> List[List[int]]:
        # Find the last non-pad token position
        last_non_pad_positions = [item - 1 for item in pad_start_positions]

        # Get the logits of the last non-pad tokens
        if logits.shape[1] == 1:
            last_logits = logits[:, 0, :]
        else:
            last_logits: List[torch.Tensor] = []
            for b_idx, last_non_pad_position in enumerate(last_non_pad_positions):
                last_logits.append(logits[b_idx, last_non_pad_position, :])
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
        flat_candidate_ids = [
            token_id
            for candidate_list in topk_candidates
            for token_id in candidate_list
        ]

        # Batch decode and strip whitespace from each token.
        decoded_tokens = [
            token.strip() for token in self.tokenizer.batch_decode(flat_candidate_ids)
        ]

        # Compute cumulative indices to split the flat list back into batches.
        sizes = [len(candidate_list) for candidate_list in topk_candidates]
        indices = [0] + list(itertools.accumulate(sizes))

        # For each batch, select the first candidate whose decoded token is not a stopword.
        return [
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


class NextWordPredictorForGPT(NextWordPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, *args, **kwargs) -> List[str]:
        super().predict(*args, **kwargs)


class NextWordPredictorForLlama(NextWordPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, *args, **kwargs) -> List[str]:
        super().predict(*args, **kwargs)


class NextWordPredictorForReLlama(NextWordPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, *args, **kwargs) -> List[str]:
        super().predict(*args, **kwargs)
