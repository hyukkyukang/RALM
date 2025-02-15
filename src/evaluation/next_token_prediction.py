from typing import *

import torch

from src.tokenization.utils import INVALID_TOKEN_ID


@torch.no_grad()
def evaluate_next_token_prediction(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
) -> Tuple[float, int]:
    """
    Evaluate model's next token prediction performance.

    Args:
        token_ids: Input token IDs tensor
        attention_mask: Attention mask tensor
        labels: Ground truth labels tensor
        model: The model to evaluate
        total_chars_cnt: Total number of characters in the dataset

    Returns:
        Tuple of (perplexity, bpb)
    """
    # Forward pass
    outputs = model(input_ids=token_ids, attention_mask=attention_mask, labels=labels)

    # Get the loss
    neg_log_likelihood_loss = outputs.loss

    # Get the number of valid tokens
    num_valid_tokens = (labels != INVALID_TOKEN_ID).sum().item()
    # Subtract the batch size due to internal label shift
    num_valid_tokens -= labels.size(0)

    # Get the loss sum
    loss_sum = neg_log_likelihood_loss * num_valid_tokens

    return loss_sum, num_valid_tokens


@torch.no_grad()
def compute_perplexity_and_bpb(
    total_loss_sum: torch.Tensor,
    total_valid_tokens_cnt: torch.Tensor,
    total_chars_cnt: torch.Tensor,
) -> Tuple[float, float]:
    # Calculate the perplexity
    perplexity = torch.exp(total_loss_sum / total_valid_tokens_cnt)
    # Use a constant for log(2) since it doesn't need to be recomputed
    LOG2 = torch.log(torch.tensor(2.0, device=total_loss_sum.device))
    # Calculate the bits-per-byte
    bpb = total_loss_sum / LOG2 / total_chars_cnt
    return (perplexity.item(), bpb.item())
