import logging
import math
from typing import *

import hkkang_utils.misc as misc_utils
import hkkang_utils.time as time_utils
import hydra
import torch
import tqdm
from omegaconf import DictConfig
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from src.model.rellama.mask import (
    generate_causal_mask_mod,
    generate_causal_retrieval_mask_mod,
)

logger = logging.getLogger("ATTENTION")

torch._dynamo.config.cache_size_limit = 1000

if torch.cuda.get_device_capability() >= (8, 0):
    torch.set_float32_matmul_precision("high")

NUM_RUNS = 1000  # Number of times each test runs
TOLERANCE = 1e-5


def get_causal_block_mask(
    input_length: int,
    kv_with_retrieval_length: int,
    device: torch.device = torch.device("cpu"),
) -> Callable:
    causal_mask_mod = generate_causal_mask_mod(
        query_length=input_length,
    )

    block_mask = create_block_mask(
        causal_mask_mod,
        1,
        1,
        input_length,
        kv_with_retrieval_length,
        device=device,
    )

    return block_mask


def get_causal_retrieval_block_mask(
    input_length: int,
    retrieval_block_num: int,
    kv_with_retrieval_length: int,
    input_chunk_size: int = 64,
    retrieval_block_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> Callable:

    causal_retrieval_mask_mod = generate_causal_retrieval_mask_mod(
        input_length=input_length,
        retrieval_block_num=retrieval_block_num,
        input_chunk_size=input_chunk_size,
        retrieval_block_size=retrieval_block_size,
        device=device,
    )

    block_mask = create_block_mask(
        causal_retrieval_mask_mod,
        1,
        1,
        input_length,
        kv_with_retrieval_length,
        device=device,
    )

    return block_mask


def flash_attention_2(query, key, value, scaling):
    """Implements FlashAttention-2"""
    query_length = query.shape[2]
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    return _flash_attention_forward(
        query,
        key,
        value,
        attention_mask=None,
        query_length=query_length,
        is_causal=True,
        softmax_scale=scaling,
        target_dtype=torch.float16,
    )


def torch_causal_attention(query_states, key_states, value_states):
    """
    Computes standard causal attention using scaled dot-product attention.
    Ensures key/value heads match query heads using Grouped Query Attention (GQA).

    This implementation mimics PyTorch's SDPA behavior by repeating key/value
    heads in block order using repeat_interleave.
    """
    bsz, nhead, q_len, head_dim = query_states.shape
    _, kv_nhead, k_len, _ = key_states.shape

    # Handle case where kv_nhead != nhead (GQA)
    if kv_nhead != nhead:
        if nhead % kv_nhead != 0:
            raise ValueError("nhead must be a multiple of kv_nhead for GQA.")
        repeat_factor = nhead // kv_nhead
        # Repeat key and value in block order ([k0, k0, k1, k1, ...])
        key_states = key_states.repeat_interleave(repeat_factor, dim=1)
        value_states = value_states.repeat_interleave(repeat_factor, dim=1)

    return torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states, attn_mask=None, is_causal=True
    )


def custom_causal_attention(query, key, value, scale):
    """Implements standard causal attention with proper GQA support.

    For GQA, if key/value heads (kv_nhead) are fewer than query heads,
    we repeat each key/value head in block order.
    This matches the behavior of PyTorch's repeat_kv:
      (batch, kv_nhead, seq_len, head_dim) -> (batch, kv_nhead * repeat_factor, seq_len, head_dim)
    """
    bsz, nhead, q_len, head_dim = query.shape
    _, kv_nhead, k_len, _ = key.shape

    if kv_nhead != nhead:
        if nhead % kv_nhead != 0:
            raise ValueError(
                "Number of query heads must be a multiple of key/value heads for GQA."
            )
        repeat_factor = nhead // kv_nhead
        # Repeat key and value in block order like [k0, k0, k1, k1, ...]
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    # Compute scaled dot-product attention
    qk = torch.einsum("bhqd, bhkd -> bhqk", query, key) * scale

    # Create a causal (lower-triangular) mask
    causal_mask = torch.tril(torch.ones((q_len, k_len), device=qk.device)).bool()
    qk.masked_fill_(~causal_mask, float("-inf"))
    attn_weights = torch.nn.functional.softmax(qk, dim=-1)

    return torch.einsum("bhqk, bhkd -> bhqd", attn_weights, value)


def compare_speed_with_and_without_torch_compile(
    prefix: str, f1, *args, **kwargs
) -> torch.Tensor:
    """Run benchmark 100 times and get the average execution time for both compiled and non-compiled functions."""
    f2 = torch.compile(f1)

    # Warm-up
    output1 = f1(*args, **kwargs)
    output2 = f2(*args, **kwargs)
    assert torch.allclose(output1, output2, atol=1e-5)

    # Measure without compilation
    time_no_compile = []
    for _ in tqdm.tqdm(range(NUM_RUNS), desc=f"Running {prefix} without torch compile"):
        t = time_utils.Timer(f"{prefix} without torch compile")
        with t.measure(False):
            f1(*args, **kwargs)
        time_no_compile.append(t.elapsed_time)

    # Measure with compilation
    time_with_compile = []
    for _ in tqdm.tqdm(range(NUM_RUNS), desc=f"Running {prefix} with torch compile"):
        t = time_utils.Timer(f"{prefix} with torch compile")
        with t.measure(False):
            f2(*args, **kwargs)
        time_with_compile.append(t.elapsed_time)

    avg_time_no_compile = sum(time_no_compile) / NUM_RUNS
    avg_time_with_compile = sum(time_with_compile) / NUM_RUNS

    logger.info(
        f"{prefix} - Avg time without compile: {avg_time_no_compile:.5f} sec | "
        f"Avg time with compile: {avg_time_with_compile:.5f} sec"
    )
    return output1


def find_differing_elements(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    tolerance: float = TOLERANCE,
    max_differences: int = 10,
) -> list[Dict[str, Any]]:
    """Find indices where two tensors differ beyond the specified tolerance."""
    diff_mask = ~torch.isclose(tensor1, tensor2, atol=tolerance)
    if not diff_mask.any():
        return None

    # Get indices of differences
    diff_indices = diff_mask.nonzero(as_tuple=True)

    # Get values at those indices
    values1 = tensor1[diff_indices]
    values2 = tensor2[diff_indices]

    # Calculate absolute differences
    abs_diffs = torch.abs(values1 - values2)

    # Sort by largest difference
    sorted_indices = torch.argsort(abs_diffs, descending=True)

    # Limit to max_differences
    if len(sorted_indices) > max_differences:
        sorted_indices = sorted_indices[:max_differences]

    results = []
    for i in sorted_indices:
        idx = tuple(dim_indices[i].item() for dim_indices in diff_indices)
        results.append(
            {
                "index": idx,
                "value1": values1[i].item(),
                "value2": values2[i].item(),
                "abs_diff": abs_diffs[i].item(),
            }
        )

    return results


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Configs
    # Model architecture parameters
    bsize = 1
    total_dim = cfg.model.architecture.hidden_size
    nhead = cfg.model.architecture.num_attention_heads
    kv_nhead = cfg.model.architecture.num_key_value_heads
    head_dim = total_dim // nhead
    enable_gqa = kv_nhead != nhead  # Group Query Attention

    # Input and scaling parameters
    input_length = cfg.model.max_length
    chunk_size = cfg.model.input_chunk_size
    scale = head_dim**-0.5

    # Retrieval parameters
    num_chunks_per_block = cfg.model.retrieval_chunk_num
    retrieval_block_size = chunk_size * num_chunks_per_block

    # Calculate number of retrieval blocks needed
    num_block_per_input = math.ceil(input_length / chunk_size) - 1

    # Calculate total sequence length including retrieval blocks
    retrieval_block_len = retrieval_block_size * num_block_per_input
    kv_with_retrieval_length = input_length + retrieval_block_len
    print(
        f"input_length: {input_length}, kv_with_retrieval_length: {kv_with_retrieval_length}"
    )

    # Device
    device = torch.device("cuda")
    dtype = torch.float32

    query = torch.randn(
        bsize,
        nhead,
        input_length,
        head_dim,
        device=device,
        requires_grad=True,
        dtype=dtype,
    )
    key = torch.randn(
        bsize,
        kv_nhead,
        kv_with_retrieval_length,
        head_dim,
        device=device,
    )
    value = torch.randn(
        bsize,
        kv_nhead,
        kv_with_retrieval_length,
        head_dim,
        device=device,
    )
    causal_retrieval_block_mask = get_causal_retrieval_block_mask(
        input_length=input_length,
        retrieval_block_num=num_block_per_input,
        kv_with_retrieval_length=kv_with_retrieval_length,
        input_chunk_size=chunk_size,
        retrieval_block_size=retrieval_block_size,
        device=device,
    )
    causal_mask = get_causal_block_mask(
        input_length=input_length,
        kv_with_retrieval_length=kv_with_retrieval_length,
        device=device,
    )

    # Compare speeds with and without torch.compile
    flex_attention_without_block_mask = compare_speed_with_and_without_torch_compile(
        "Flex attention without block mask",
        flex_attention,
        query,
        key,
        value,
        block_mask=None,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=False,
    )

    flex_attention_with_causal_retrieval_block_mask = (
        compare_speed_with_and_without_torch_compile(
            "Flex attention with causal retrieval block mask",
            flex_attention,
            query,
            key,
            value,
            block_mask=causal_retrieval_block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=False,
        )
    )

    flex_attention_with_causal_mask = compare_speed_with_and_without_torch_compile(
        "Flex attention with causal mask",
        flex_attention,
        query,
        key,
        value,
        block_mask=causal_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=False,
    )

    custom_causal_attention_output = compare_speed_with_and_without_torch_compile(
        "Custom causal attention",
        custom_causal_attention,
        query,
        key,
        value,
        scale=scale,
    )

    torch_causal_attention_output = compare_speed_with_and_without_torch_compile(
        "Torch causal attention",
        torch_causal_attention,
        query,
        key,
        value,
    )

    # Run flash attention-2 only if the GPU compatibility is above 8.0
    if torch.cuda.get_device_capability() >= (8, 0):
        flash_attention_2_output = compare_speed_with_and_without_torch_compile(
            "FlashAttention-2",
            flash_attention_2,
            query,
            key,
            value,
            scaling=scale,
        )

    # Check the results are very close to each other
    assert not torch.allclose(
        flex_attention_without_block_mask,
        flex_attention_with_causal_retrieval_block_mask,
        atol=TOLERANCE,
    ), f"Flex attention without block mask and with causal retrieval block mask are close to each other, which should not happen"
    assert torch.allclose(
        torch_causal_attention_output,
        custom_causal_attention_output,
        atol=TOLERANCE,
    ), f"Torch causal attention and custom causal attention are not close to each other"

    there_is_retrieved_chunks = retrieval_block_len > 0

    # Check if attention outputs should match based on whether there are retrieved chunks
    should_match = not there_is_retrieved_chunks
    are_close = torch.allclose(
        flex_attention_with_causal_retrieval_block_mask,
        flex_attention_with_causal_mask,
        atol=TOLERANCE,
    )
    # Assert based on expected behavior
    assert are_close == should_match, (
        f"Flex attention with causal retrieval block mask and with causal mask "
        f"{'should be' if should_match else 'should not be'} close to each other"
    )

    assert torch.allclose(
        flex_attention_with_causal_mask,
        custom_causal_attention_output,
        atol=TOLERANCE,
    ), f"Flex attention with causal mask and custom causal attention are not close to each other"
    assert torch.allclose(
        flex_attention_with_causal_mask,
        torch_causal_attention_output,
        atol=TOLERANCE,
    ), f"Flex attention with causal mask and torch causal attention are not close to each other"
    if torch.cuda.get_device_capability() >= (8, 0):
        assert torch.allclose(
            flex_attention_with_causal_mask,
            flash_attention_2_output,
            atol=TOLERANCE,
        ), f"Flex attention with causal mask and flash attention-2 are not close to each other"

    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
