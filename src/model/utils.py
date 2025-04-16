import logging
import math
from typing import *

import torch
from tensordict import TensorDict
from transformers.cache_utils import DynamicCache
from calflops import calculate_flops
from src.dataset.utils import batch_step_to_position
from src.utils import is_torch_compile_possible, log_if_rank_zero
from src.tokenization import ReLlamaTokenizer

logger = logging.getLogger("ModelUtils")

MODEL_STATE_DICT_KEY = "state_dict"
TORCH_COMPILE_MODULE_KEY = "_orig_mod."


def remove_prefix(text, substring: str) -> str:
    if substring in text:
        return text.replace(substring, "")
    return text


def add_prefix(text, src_text: str, dst_text: str) -> str:
    if text.startswith(src_text):
        return dst_text + text[len(src_text) :]
    return text


def convert_checkpoint_for_evaluation(path: str) -> None:
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


def convert_checkpoint_for_training(path: str) -> None:
    # Load the checkpoint
    ckpt = torch.load(path, weights_only=False, map_location="cpu")

    # Replace "model.model." to "model._orig_mod.model."
    src_text = "model."
    dst_text = "model._orig_mod."
    in_state_dict = ckpt[MODEL_STATE_DICT_KEY]
    pairings = [
        (src_key, add_prefix(src_key, src_text, dst_text))
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

    # Make use_torch_compile to True
    ckpt["hyper_parameters"]["use_torch_compile"] = True

    # Save the checkpoint
    log_if_rank_zero(logger, f"Saving checkpoint to {path}")
    torch.save(ckpt, path)


def update_batch_step_in_checkpoint_to_consider_gradient_accumulation(
    checkpoint: Dict[str, Any], gradient_accumulation_steps: int
) -> Dict[str, Any]:
    """
    Update the batch step in the checkpoint to consider gradient accumulation.
    """

    def modify_step(step: int) -> int:
        remainder = step % gradient_accumulation_steps
        return step - remainder

    # Modify the batches_that_stepped
    checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"][
        "_batches_that_stepped"
    ] = modify_step(
        checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"][
            "_batches_that_stepped"
        ]
    )
    for key in ["total", "current"]:
        for k, v in checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"][
            key
        ].items():
            checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"][key][k] = (
                modify_step(v)
            )
    return checkpoint


def update_position_in_checkpoint_for_consistency(
    checkpoint: Dict[str, Any], per_device_batch_size: int, num_devices: int
) -> Dict[str, Any]:
    """
    Update the position in the checkpoint for consistency.
    """
    current_batch_step = (
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
            "completed"
        ]
        + 1
    )
    position = batch_step_to_position(current_batch_step, per_device_batch_size)

    # Update the position of the sampler
    checkpoint["loops"]["fit_loop"]["state_dict"]["combined_loader"][0][
        "position"
    ] = position
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
    if use_compile and is_torch_compile_possible():
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
    min_lr: float = 0.0,
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
        progress = float(it - (warmup_iters + intermediate_iters)) / float(
            total_iters - (warmup_iters + intermediate_iters)
        )
        return (intermediate_lr + (min_lr - intermediate_lr) * progress) / max_lr
    else:
        return min_lr / max_lr


def update_dynamic_cache(
    dynamic_cache: DynamicCache,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    pad_start_positions: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

    Parameters:
        key_states (`torch.Tensor`):
            The new key states to cache.
        value_states (`torch.Tensor`):
            The new value states to cache.
        layer_idx (`int`):
            The index of the layer to cache the states for.
        cache_kwargs (`Dict[str, Any]`, `optional`):
            Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

    Return:
        A tuple containing the updated key and value states.
    """

    # Update the number of seen tokens
    if layer_idx == 0:
        dynamic_cache._seen_tokens += key_states.shape[-2]

    # Update the cache
    if key_states is not None:
        if len(dynamic_cache.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(dynamic_cache.key_cache), layer_idx):
                dynamic_cache.key_cache.append([])
                dynamic_cache.value_cache.append([])
            dynamic_cache.key_cache.append(key_states)
            dynamic_cache.value_cache.append(value_states)
        elif len(dynamic_cache.key_cache[layer_idx]) == 0:
            # fills previously skipped layers; checking for tensor causes errors
            dynamic_cache.key_cache[layer_idx] = key_states
            dynamic_cache.value_cache[layer_idx] = value_states
        else:
            if pad_start_positions is None:
                dynamic_cache.key_cache[layer_idx] = torch.cat(
                    [dynamic_cache.key_cache[layer_idx], key_states], dim=-2
                )
                dynamic_cache.value_cache[layer_idx] = torch.cat(
                    [dynamic_cache.value_cache[layer_idx], value_states], dim=-2
                )
            else:
                # Insert the new key and value states starting at the non-pad positions
                new_key_cache: List[torch.Tensor] = []
                new_value_cache: List[torch.Tensor] = []
                for batch_idx, pad_start_idx in enumerate(pad_start_positions):
                    # Insert the current query at the position before the non-pad token
                    insert_position = pad_start_idx - 1
                    
                    # Concatenate the new key and value states with the old key and value states
                    new_key_cache.append(
                        torch.cat(
                            [
                                dynamic_cache.key_cache[layer_idx][
                                    batch_idx, :, :insert_position
                                ],
                                key_states[batch_idx, :, :],
                                dynamic_cache.key_cache[layer_idx][
                                    batch_idx, :, insert_position:
                                ],
                            ],
                            dim=-2,
                        )
                    )
                    new_value_cache.append(
                        torch.cat(
                            [
                                dynamic_cache.value_cache[layer_idx][
                                    batch_idx, :, :insert_position
                                ],
                                value_states[batch_idx, :, :],
                                dynamic_cache.value_cache[layer_idx][
                                    batch_idx, :, insert_position:
                                ],
                            ],
                            dim=-2,
                        )
                    )
                dynamic_cache.key_cache[layer_idx] = torch.stack(new_key_cache)
                dynamic_cache.value_cache[layer_idx] = torch.stack(new_value_cache)

    return dynamic_cache.key_cache[layer_idx], dynamic_cache.value_cache[layer_idx]


def calculate_FLOPs(model: torch.nn.Module, tokenizer: ReLlamaTokenizer, max_seq_len: int) -> int:
    flops, macs, params = calculate_flops(model=model, 
                                        input_shape=(1, max_seq_len),
                                        transformer_tokenizer=tokenizer,
                                        include_backPropagation=True,
                                        print_results=False,
                                        print_detailed=False,
                                        output_as_string=False)
    return int(flops)