import math

import torch


# Custom weight initialization function
def initialize_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
        # Initialize weights for Linear and Conv1d layers
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        # Initialize weights for Embedding layers
        torch.nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)


# Define a TorchScript-compatible function for the learning rate schedule
@torch.jit.script
def lr_lambda_cosine_decay(
    it: int, warmup_iters: int, total_iters: int, max_lr: float, min_lr: float
) -> float:
    """
    Computes learning rate scaling factor for warmup and cosine decay.

    Args:
        it (int): Current optimizer step.
        warmup_iters (int): Number of warmup steps.
        total_iters (int): Total optimizer steps.
        min_lr (float): Minimum learning rate.
        max_lr (float): Maximum learning rate.

    Returns:
        float: The learning rate multiplier.
    """
    if it < warmup_iters:
        return float(it) / float(warmup_iters)  # Linear warmup
    elif it >= total_iters:
        return min_lr / max_lr  # Hold at min LR
    else:
        decay_ratio = float(it - warmup_iters) / float(total_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine decay
        # Ensure we don't go below min_lr
        return max(min_lr / max_lr, (min_lr + coeff * (max_lr - min_lr)) / max_lr)


@torch.jit.script
def lr_lambda_linear_decay(
    it: int, warmup_iters: int, total_iters: int, max_lr: float, min_lr: float = 0.0
) -> float:
    """
    Computes learning rate scaling factor for warmup and linear decay to zero.

    Args:
        it (int): Current optimizer step.
        warmup_iters (int): Number of warmup steps.
        total_iters (int): Total optimizer steps.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.

    Returns:
        float: The learning rate multiplier.
    """
    if it < warmup_iters:
        return float(it) / float(warmup_iters)  # Linear warmup
    elif it >= total_iters:
        return min_lr / max_lr
    else:
        # Linear decay to min_lr, normalized by max_lr for use with LambdaLR
        return (
            min_lr
            + (max_lr - min_lr) * (total_iters - it) / (total_iters - warmup_iters)
        ) / max_lr
