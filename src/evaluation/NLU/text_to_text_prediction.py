import logging
from typing import *

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("TextToTextPrediction")


@torch.no_grad()
def predict_next_tokens_from_choices(
    batch_context_token_ids: torch.Tensor,
    token_ids_of_choices: List[List[int]],
    model: AutoModelForCausalLM,
    retrieved_chunk_ids: Optional[torch.Tensor] = None,
) -> List[int]:
    """Give continuation of the line with at most max_predictions BPE tokens. Returns line extended with predictions of
    the model."""
    selected_token_ids: List[int] = []
    batch_context_token_ids = torch.tensor(batch_context_token_ids, device=model.device)
    for bidx, token_ids in enumerate(batch_context_token_ids):
        # Truncate the token ids to the max length
        if token_ids.shape[0] > model.config.max_position_embeddings:
            logger.warning(
                f"Truncating the token ids to the max length: {token_ids.shape[0]} -> {model.config.max_position_embeddings}"
            )
            token_ids = token_ids[: model.config.max_position_embeddings]
        outputs = model(token_ids.unsqueeze(0), retrieved_chunk_ids=retrieved_chunk_ids)
        logits: torch.Tensor = outputs.logits[0, -1]  # Get logits from the outputs
        # Get the logits for the choices
        choice_logits: torch.Tensor = logits[token_ids_of_choices[bidx]]
        # Find the choice with the highest logit
        choice_idx: int = torch.argmax(choice_logits).item()
        # Get the token id of the choice
        choice_token_id: int = token_ids_of_choices[bidx][choice_idx]
        selected_token_ids.append(choice_token_id)

    return selected_token_ids


@torch.no_grad()
def evaluate_text_to_text_prediction(
    batch_token_ids: torch.Tensor,
    target_texts: List[str],
    text_choices: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    retrieved_chunk_ids: Optional[torch.Tensor] = None,
    is_analyze: bool = False,
) -> List[bool]:
    """batch_token_ids shape: (bsize, seq_len)
    There should be no padding tokens in the batch_token_ids, which means all the sequences should be of the same length.
    """
    bsize: int = len(batch_token_ids)
    # Preprocess input token ids
    # Remove the last token from the input token ids
    batch_context_token_ids: List[List[int]] = []
    batch_target_token_id: List[int] = []
    batch_choice_token_ids: List[List[int]] = []
    for token_ids, target_text in zip(batch_token_ids, target_texts):
        context_token_ids, target_token_id, other_choice_token_ids = (
            extract_context_target_and_choices(
                token_ids=token_ids,
                target_text=target_text,
                choices=text_choices,
                tokenizer=tokenizer,
            )
        )
        batch_context_token_ids.append(context_token_ids)
        batch_target_token_id.append(target_token_id)
        batch_choice_token_ids.append([target_token_id] + other_choice_token_ids)

    # Get the predicted completions
    selected_token_ids: List[int] = predict_next_tokens_from_choices(
        batch_context_token_ids=batch_context_token_ids,
        token_ids_of_choices=batch_choice_token_ids,
        model=model,
        retrieved_chunk_ids=retrieved_chunk_ids,
    )

    # Check if the predicted token ids are the same as the target token ids
    batch_is_correct: List[bool] = [
        selected_token_ids[idx] == batch_target_token_id[idx] for idx in range(bsize)
    ]

    # Check if the predicted word is the same as the last word
    if is_analyze:
        pass
    return batch_is_correct


def extract_context_target_and_choices(
    token_ids: List[int],
    target_text: str,
    choices: List[str],
    tokenizer: AutoTokenizer,
) -> Tuple[List[int], List[int], List[int]]:
    """Split the token ids into context and target."""
    # Find the number of tokens for the target token
    num_tokens_for_target_token_wo_space: int = len(
        tokenizer.encode(target_text, add_special_tokens=False)
    )
    num_tokens_for_target_token_w_space: int = len(
        tokenizer.encode(" " + target_text, add_special_tokens=False)
    )
    num_tokens_for_target_token: int = min(
        num_tokens_for_target_token_wo_space, num_tokens_for_target_token_w_space
    )
    # Find the context token ids
    context_token_ids: List[int] = token_ids[:-num_tokens_for_target_token].tolist()
    # Find the target token ids
    target_token_ids: List[int] = token_ids[-num_tokens_for_target_token:].tolist()
    target_token_id: int = target_token_ids[0]
    # Decode the context and target token ids
    context_text: str = tokenizer.decode(context_token_ids, skip_special_tokens=True)
    decoded_target_text: str = tokenizer.decode(
        target_token_ids, skip_special_tokens=True
    )

    # Check the target text is the same as the decoded target text
    assert (
        target_text in decoded_target_text
    ), f"{target_text} is not sub-string of {decoded_target_text}"

    # Find the alternative token ids for the target text
    choice_token_ids: List[int] = []
    for choice in choices:
        if choice != target_text:
            # Replace the target text with the choice
            new_texts = context_text + decoded_target_text.replace(target_text, choice)
            # Tokenize the new texts
            new_token_ids = tokenizer.encode(new_texts)
            new_token_ids = new_token_ids[: len(context_token_ids) + 1]
            # Check the context part is the same
            decoded_new_token_ids = tokenizer.decode(new_token_ids[:-1])
            decoded_context_token_ids = tokenizer.decode(context_token_ids)
            assert (
                decoded_new_token_ids == decoded_context_token_ids
            ), f"Context part is not the same: {decoded_new_token_ids} != {decoded_context_token_ids}"

            # Append the choice token ids
            choice_token_ids.append(new_token_ids[-1])

    # Check that the choice token ids are all different
    # If not, we need to generate more than one tokens
    possible_choices: List[int] = [target_token_id] + choice_token_ids
    assert len(possible_choices) == len(
        set(possible_choices)
    ), "Choice token ids are not all different"

    return context_token_ids, target_token_id, choice_token_ids
