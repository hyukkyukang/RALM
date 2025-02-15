import copy
import logging
from typing import *

import lightning as L
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset.utils import SingletonBasicTokenizer
from src.evaluation.utils import STOPWORDS_FROM_GPT2
from src.utils import log_if_rank_zero

logger = logging.getLogger("NextWordPrediction")


def is_model_compiled(model: Union[L.LightningModule, AutoModelForCausalLM]) -> bool:
    if isinstance(model, L.LightningModule):
        if (
            "use_torch_compile" in model.cfg
            and model.cfg.use_torch_compile
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 7
        ):
            assert isinstance(
                model.model, torch._dynamo.eval_frame.OptimizedModule
            ), f"Model is not an OptimizedModule?: {type(model.model)}"
            return True
    else:
        return isinstance(model, torch._dynamo.eval_frame.OptimizedModule)


@torch.no_grad()
def predict_next_tokens(
    batch_token_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    steps_to_predict: int = 6,
    beam_width: int = 128,
) -> List[str]:
    """Give continuation of the line with at most max_predictions BPE tokens. Returns line extended with predictions of
    the model."""
    bsize = batch_token_ids.size(0)

    # List to append the predictions
    all_token_ids: List[List[int]] = batch_token_ids.tolist()

    # Perform token prediction step by step
    current_input_token_ids = batch_token_ids
    states = None
    for _ in range(steps_to_predict):
        outputs = model(current_input_token_ids, past_key_values=states)
        logits = outputs.logits  # Get logits from the outputs
        # Clone if using torch.compile
        if is_model_compiled(model):
            # states = outputs.past_key_values.clone()  # Get the state from outputs
            states = copy.deepcopy(outputs.past_key_values)

        else:
            states = outputs.past_key_values

        # Get the top k candidates
        _, line_encoded_candidates = torch.topk(
            logits[:, -1, :],
            k=beam_width,
            dim=-1,
        )
        line_encoded_candidates: List[List[int]] = line_encoded_candidates.tolist()
        # Find the token with the highest probability and not a stopword
        current_input_token_ids: List[List[int]] = [[] for _ in range(bsize)]
        for b_idx in range(bsize):
            predicted_token_id = None
            for candidate_token_id in line_encoded_candidates[b_idx]:
                candidate_token = tokenizer.decode(candidate_token_id).strip()
                if candidate_token not in STOPWORDS_FROM_GPT2:
                    # Select this candidate token as the predicted token
                    predicted_token_id = candidate_token_id
                    break
            assert predicted_token_id is not None, "No valid candidate found"
            # Append for next step
            current_input_token_ids[b_idx].append(predicted_token_id)
            # Append for final output containing text from all steps
            all_token_ids[b_idx].append(predicted_token_id)
        # Cast the input token ids to a tensor
        current_input_token_ids = torch.tensor(current_input_token_ids).to(model.device)

    # Convert the decoded sequences to a list of strings
    return [tokenizer.decode(token_ids) for token_ids in all_token_ids]


@torch.no_grad()
def evaluate_last_word_prediction(
    batch_token_ids: torch.Tensor,
    target_last_words: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    is_analyze: bool = False,
) -> List[bool]:
    """batch_token_ids shape: (bsize, seq_len)
    There should be no padding tokens in the batch_token_ids, which means all the sequences should be of the same length.
    """
    basic_tokenizer = SingletonBasicTokenizer()
    # Get the predicted completions
    batch_text_with_predictions: List[str] = predict_next_tokens(
        batch_token_ids=batch_token_ids,
        tokenizer=tokenizer,
        model=model,
    )
    batch_input_contexts: List[str] = [
        tokenizer.decode(token_ids) for token_ids in batch_token_ids
    ]
    batch_generated_texts: List[str] = [
        text_with_predictions[len(input_contexts) :].strip()
        for text_with_predictions, input_contexts in zip(
            batch_text_with_predictions, batch_input_contexts
        )
    ]
    batch_predicted_words: List[List[str]] = [
        basic_tokenizer.tokenize(generated_texts)
        for generated_texts in batch_generated_texts
    ]
    # Handle the case where the predicted words are empty
    for idx, predicted_words in enumerate(batch_predicted_words):
        if not predicted_words:
            batch_predicted_words[idx] = [""]
    # Compare the predicted words with the target last words
    batch_is_correct: List[bool] = [
        predicted_words[0] == last_word
        for predicted_words, last_word in zip(batch_predicted_words, target_last_words)
    ]
    # Check if the predicted word is the same as the last word
    if is_analyze:
        analyze_idx = 0
        log_if_rank_zero(logger, f"Input context: {batch_input_contexts[analyze_idx]}")
        log_if_rank_zero(
            logger, f"Generated text: {batch_generated_texts[analyze_idx]}"
        )
        log_if_rank_zero(
            logger, f"Predicted words: {batch_predicted_words[analyze_idx]}"
        )
        log_if_rank_zero(logger, f"Target word: {target_last_words[analyze_idx]}")
        log_if_rank_zero(logger, f"Is correct: {batch_is_correct[analyze_idx]}")
        log_if_rank_zero(logger, "-" * 100 + "\n")
    return batch_is_correct
