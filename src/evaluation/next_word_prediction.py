import copy
import logging
from typing import *

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset.utils import SingletonBasicTokenizer
from src.evaluation.utils import STOPWORDS_FROM_GPT2
from src.utils import log_if_rank_zero

logger = logging.getLogger("NextWordPrediction")


@torch.no_grad()
def predict_next_tokens(
    token_ids: List[int],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    steps_to_predict: int = 6,
    beam_width: int = 128,
) -> str:
    """Give continuation of the line with at most max_predictions BPE tokens. Returns line extended with predictions of
    the model."""
    # Convert the token ids to a tensor
    current_input_token_ids = torch.tensor(token_ids).to(model.device)
    # Get the predictions
    all_token_ids: List[int] = copy.deepcopy(token_ids)
    states = None
    for _ in range(steps_to_predict):
        outputs = model(current_input_token_ids.unsqueeze(0), past_key_values=states)
        logits = outputs.logits  # Get logits from the outputs
        states = outputs.past_key_values  # Get the state from outputs

        # Get the top k candidates
        _, line_encoded_candidates = torch.topk(
            logits[0, -1, :],
            k=beam_width,
            dim=-1,
        )
        line_encoded_candidates = line_encoded_candidates.tolist()
        # Convert all the candidates to tokens
        candidate_tokens: List[str] = [
            tokenizer.decode(item).strip() for item in line_encoded_candidates
        ]
        # Find the first candidate which is not a stopword
        predicted_token_id = None
        for cand_idx, candidate_token in enumerate(candidate_tokens):
            if candidate_token not in STOPWORDS_FROM_GPT2:
                predicted_token_id = line_encoded_candidates[cand_idx]
                break
        assert predicted_token_id is not None, "No valid candidate found"
        all_token_ids.append(predicted_token_id)

        # Update the input tensor to pass to the next step
        current_input_token_ids = torch.tensor([predicted_token_id]).to(model.device)

    # Convert the decoded sequences to a list of strings
    return tokenizer.decode(all_token_ids)


@torch.no_grad()
def evaluate_last_word_prediction(
    token_ids: List[int],
    last_word: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    is_analyze: bool = False,
) -> bool:
    basic_tokenizer = SingletonBasicTokenizer()
    # Get the predicted completions
    text_with_predictions: str = predict_next_tokens(
        token_ids=token_ids,
        tokenizer=tokenizer,
        model=model,
    )
    input_contexts: str = tokenizer.decode(token_ids)
    generated_texts: str = text_with_predictions[len(input_contexts) :].strip()
    predicted_words: List[str] = basic_tokenizer.tokenize(generated_texts)
    predicted_word: str = "" if len(predicted_words) == 0 else predicted_words[0]
    is_correct: bool = predicted_word == last_word
    # Check if the predicted word is the same as the last word
    if is_analyze:
        log_if_rank_zero(logger, f"Input context: {input_contexts}")
        log_if_rank_zero(logger, f"Generated text: {generated_texts}")
        log_if_rank_zero(logger, f"Predicted words: {predicted_words}")
        log_if_rank_zero(logger, f"Target word: {last_word}")
        log_if_rank_zero(logger, f"Is correct: {is_correct}")
        log_if_rank_zero(logger, "-" * 100 + "\n")
    return is_correct
