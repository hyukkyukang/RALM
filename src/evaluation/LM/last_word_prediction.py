import logging
from typing import *

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset.utils import SingletonBasicTokenizer
from src.model.wrapper.next_word_predictor import NextWordPredictor
from src.utils import log_if_rank_zero

logger = logging.getLogger("LastWordPrediction")


def evaluate_last_word_prediction(
    batch_token_ids: torch.Tensor,
    target_last_words: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    pad_start_positions: Optional[torch.LongTensor] = None,
    retrieved_input_ids: Optional[torch.Tensor] = None,
    num_retrieval_blocks: Optional[List[int]] = None,
    is_analyze: bool = False,
) -> List[bool]:
    """batch_token_ids shape: (bsize, seq_len)
    There should be no padding tokens in the batch_token_ids, which means all the sequences should be of the same length.
    """
    basic_tokenizer = SingletonBasicTokenizer()

    # Get the predicted completions
    next_word_predictor = NextWordPredictor(model, tokenizer)
    batch_text_with_predictions: List[str] = next_word_predictor.predict(
        input_ids=batch_token_ids,
        pad_start_positions=pad_start_positions,
        retrieved_input_ids=retrieved_input_ids,
        num_retrieval_blocks=num_retrieval_blocks,
    )

    # Remove the padding tokens
    batch_token_ids_wo_padding: List[List[int]] = [
        token_ids[: pad_start_positions[idx]]
        for idx, token_ids in enumerate(batch_token_ids.tolist())
    ]

    # Postprocess the predicted completions
    batch_input_contexts: List[str] = tokenizer.batch_decode(batch_token_ids_wo_padding)
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
        for analyze_idx in range(len(batch_input_contexts)):
            log_if_rank_zero(
                logger, f"Input context: {batch_input_contexts[analyze_idx]}"
            )
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
