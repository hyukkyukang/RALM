import logging
import math
from typing import *

import torch
from tensordict import TensorDict

from src.dataset.utils import batch_step_to_position
from src.utils import is_torch_compile_possible, log_if_rank_zero

logger = logging.getLogger("ModelUtils")

MODEL_STATE_DICT_KEY = "state_dict"
TORCH_COMPILE_MODULE_KEY = "_orig_mod."


def remove_prefix(text, substring: str) -> str:
    if substring in text:
        return text.replace(substring, "")
    return text


def repair_checkpoint(path):
    # Load the checkpoint
    ckpt = torch.load(path, weights_only=False, map_location="cpu")

    # Repair the checkpoint
    in_state_dict = ckpt[MODEL_STATE_DICT_KEY]
    pairings = [
        (src_key, remove_prefix(src_key, TORCH_COMPILE_MODULE_KEY))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        log_if_rank_zero(logger, f"No need to repair checkpoint {path}")
        return  # Do not write checkpoint if no need to repair!
    out_state_dict = {}
    for src_key, dest_key in pairings:
        log_if_rank_zero(logger, f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]
    ckpt[MODEL_STATE_DICT_KEY] = out_state_dict

    # Make use_torch_compile to False
    ckpt["hyper_parameters"]["use_torch_compile"] = False

    # Save the checkpoint
    log_if_rank_zero(logger, f"Saving checkpoint to {path}")
    torch.save(ckpt, path)


def update_batch_step_in_checkpoint_to_consider_gradient_accumulation(checkpoint: Dict[str, Any], 
                                                                      gradient_accumulation_steps: int) -> Dict[str, Any]:
    """
    Update the batch step in the checkpoint to consider gradient accumulation.
    """
    batches_that_stepped = checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"]["_batches_that_stepped"]
    value_to_subtract = batches_that_stepped % gradient_accumulation_steps
    new_batches_that_stepped = batches_that_stepped - value_to_subtract
    checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"]["_batches_that_stepped"] = new_batches_that_stepped
    for key in ["total", "current"]:
        for k, v in checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"][key].items():
            checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"][key][k] = new_batches_that_stepped
            # print(f"Updating {key} {k} from {v} to {checkpoint['loops']['fit_loop']['epoch_loop.batch_progress'][key][k]}")
    return checkpoint

def update_position_in_checkpoint_for_consistency(checkpoint: Dict[str, Any], 
                                                  per_device_batch_size: int) -> Dict[str, Any]:
    """
    Update the position in the checkpoint for consistency.
    """
    batches_that_stepped = checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"]["_batches_that_stepped"]
    position = batch_step_to_position(batches_that_stepped+1, per_device_batch_size)
    checkpoint["loops"]["fit_loop"]["state_dict"]["combined_loader"][0]["position"] = position
    return checkpoint

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


def get_compile_decorator(
    use_compile: bool = True, fullgraph: bool = False, mode: str = "default"
):
    """Returns torch.compile decorator if GPU is capable and use_compile is True, otherwise returns a no-op decorator"""
    if (
        use_compile
        and is_torch_compile_possible()
    ):
        log_if_rank_zero(
            logger, f"Compiling the module with torch compile in {mode} mode..."
        )
        return torch.compile(fullgraph=fullgraph, mode=mode)
    return lambda x: x  # no-op decorator


def add_to_tensor_dict_safely(
    tensor_dict: TensorDict, key: str, tensor: torch.Tensor
) -> TensorDict:
    """
    Add a tensor to a dictionary safely.
    """
    tensor_dict[key] = tensor_dict.get(key, torch.tensor(0)) + tensor
    return tensor_dict


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
    it: int, 
    warmup_iters: int, 
    intermediate_iters: int, 
    total_iters: int, 
    max_lr: float, 
    intermediate_lr: float, 
    min_lr: float = 0.0
) -> float:
    """
    Computes a learning rate multiplier that follows:
      - Linear warmup for `warmup_iters`
      - Linear decay from max_lr to intermediate_lr over `intermediate_iters` steps
      - Linear decay from intermediate_lr to min_lr over the remaining iterations

    Args:
        it (int): Current optimizer step.
        warmup_iters (int): Number of warmup steps.
        intermediate_iters (int): Number of iterations for decay from max_lr to intermediate_lr.
        total_iters (int): Total optimizer steps.
        max_lr (float): Maximum learning rate.
        intermediate_lr (float): Learning rate at the end of the intermediate phase.
        min_lr (float): Minimum learning rate.

    Returns:
        float: The learning rate multiplier.
    """
    if it < warmup_iters:
        # Linear warmup: factor increases from 0 to 1.
        return float(it) / float(warmup_iters)
    elif it < warmup_iters + intermediate_iters:
        # Linear decay from max_lr to intermediate_lr.
        # Normalized multiplier: at it == warmup_iters -> 1.0, at it == warmup_iters + intermediate_iters -> intermediate_lr / max_lr.
        progress = float(it - warmup_iters) / float(intermediate_iters)
        return (max_lr + (intermediate_lr - max_lr) * progress) / max_lr
    elif it < total_iters:
        # Linear decay from intermediate_lr to min_lr.
        # This phase spans from it = warmup_iters + intermediate_iters to it = total_iters.
        progress = float(it - (warmup_iters + intermediate_iters)) / float(total_iters - (warmup_iters + intermediate_iters))
        return (intermediate_lr + (min_lr - intermediate_lr) * progress) / max_lr
    else:
        return min_lr / max_lr
    