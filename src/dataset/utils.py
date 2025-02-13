import re
from typing import *

import hkkang_utils.pattern as pattern_utils
import torch
from transformers import BasicTokenizer

from src.tokenization.utils import INVALID_TOKEN_ID


@pattern_utils.singleton
class SingletonBasicTokenizer:
    def __init__(self):
        self.basic_tokenizer = BasicTokenizer()

    def tokenize(self, text: str) -> List[str]:
        return self.basic_tokenizer.tokenize(text)


@torch.jit.script
def count_avg_chars_per_token_in_batch(
    attention_masks: List[List[int]],
    full_texts: List[str],
    return_total_valid_tokens: bool = False,
) -> Union[float, Tuple[float, int]]:
    # Calculate character counts and valid tokens in one pass
    char_counts_per_seq: List[int] = [len(text) for text in full_texts]
    valid_tokens_per_seq: List[int] = [sum(mask) for mask in attention_masks]
    num_valid_tokens_total: int = sum(valid_tokens_per_seq)
    num_total_tokens_cnt: int = sum(char_counts_per_seq)

    # Calculate average characters per token
    avg_char_in_token = num_total_tokens_cnt / num_valid_tokens_total

    if return_total_valid_tokens:
        return avg_char_in_token, num_valid_tokens_total
    else:
        return avg_char_in_token


def split_text_into_context_and_last_word(line: str) -> Dict[str, str]:
    line = line.strip()
    basic_tokenizer = SingletonBasicTokenizer()
    toks = basic_tokenizer.tokenize(line)
    length_of_word = len(toks[-1])
    assert length_of_word > 0, f"The last word is empty: {toks[-1]}"
    return {"context": line[:-length_of_word].strip(), "last_word": toks[-1]}


def normalize_quotes(text: str) -> str:
    """
    Normalize various types of single and double quotes to standard quotes (' and ").
    Handles curly quotes, prime marks, and their combinations.
    Uses regex for efficient pattern matching.
    """
    # Map of quote patterns to be replaced with standard quotes
    quote_patterns = [
        (r'[""‟„″]', '"'),  # various double quotes to standard double quote
        (r"``", '"'),  # double backticks to double quote
        (r"[" "‛′]", "'"),  # various single quotes to standard single quote
        (r"`", "'"),  # single backtick to single quote
    ]

    result = text
    for pattern, replacement in quote_patterns:
        result = re.sub(pattern, replacement, result)

    return result


def perform_sliding_window_segmentation(
    token_ids: List[int], window_size: int, stride: int
) -> List[Dict[str, Any]]:
    """
    Segment token IDs into overlapping windows for training, ensuring that:
      1) The first segment is fully trainable (no -100).
      2) Intermediate segments only have 'new' tokens trainable.
      3) The *last* segment is exactly 'window_size' tokens long (forced),
         potentially overlapping the previous segment by more than `stride`.
    """

    segments: List[Dict[str, Any]] = []

    # We'll track how far we've actually covered so far.
    prev_end = 0
    current_start = 0
    n_tokens = len(token_ids)

    # Edge case: if the sequence is shorter than or equal to window_size,
    # we just create one segment that covers the entire sequence.
    if n_tokens <= window_size:
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = input_ids.clone()  # All are "new" for the single segment

        segments.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "is_sliding_window": False,
                "actual_end": n_tokens,
            }
        )
        return segments

    # While our next segment could be a *full* window without exceeding
    # the token_ids length, keep adding segments at stride increments.
    while True:
        # If we can shift a full window from current_start without
        # exceeding the total length, do so:
        if current_start + window_size < n_tokens:
            # Not the last segment yet
            start_idx = current_start
            end_idx = start_idx + window_size
            segment_tokens = token_ids[start_idx:end_idx]
        else:
            # We are about to exceed or exactly reach the end. Force
            # the last segment to be exactly 'window_size' length
            # that ENDS at n_tokens:
            start_idx = max(0, n_tokens - window_size)
            end_idx = n_tokens
            segment_tokens = token_ids[start_idx:end_idx]

        # Build tensors
        input_ids = torch.tensor(segment_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = input_ids.clone()

        # Figure out how many tokens are new:
        # For the first segment: everything is new
        if len(segments) == 0:
            num_new_tokens = len(segment_tokens)
        else:
            # new = how many indices go beyond prev_end
            # (the coverage of the last appended segment)
            num_new_tokens = end_idx - prev_end
            if num_new_tokens < 0:
                num_new_tokens = 0

        # Mask out everything except the final num_new_tokens
        seg_len = len(segment_tokens)
        if seg_len > num_new_tokens:
            labels[: seg_len - num_new_tokens] = INVALID_TOKEN_ID

        # Append
        segments.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "is_sliding_window": True,
                "actual_end": end_idx,
            }
        )

        # Update coverage
        prev_end = end_idx

        # If we just created the final forced window (end_idx == len(token_ids)), we stop
        if end_idx == n_tokens:
            break

        # Otherwise, increment the start by stride
        current_start += stride

    return segments
