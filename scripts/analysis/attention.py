import functools
import logging
import math
from typing import *

import hkkang_utils.misc as misc_utils
import hkkang_utils.time as time_utils
import hydra
import matplotlib.pyplot as plt
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

NUM_RUNS = 1000
TOLERANCE = 1e-5

if torch.cuda.get_device_capability() >= (8, 0):
    torch.set_float32_matmul_precision("high")
    # Decrease the tolerance if using precision high
    TOLERANCE = 1e-2


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


@functools.lru_cache(maxsize=None)
def create_custom_causal_retrieval_block_mask(
    q_len: int,
    k_len: int,
    chunk_size: int,
    retrieval_block_size: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create a custom causal retrieval block mask"""
    retrieval_block_len = k_len - q_len
    retrieval_block_num = retrieval_block_len // retrieval_block_size

    # Create attention mask
    causal_mask = torch.tril(
        torch.ones((q_len, q_len), device=device, dtype=torch.bool)
    )
    retrieval_block_mask = torch.zeros(
        (q_len, retrieval_block_len), device=device, dtype=torch.bool
    )
    for i in range(retrieval_block_num):
        q_start = (i + 1) * chunk_size
        q_end = q_start + chunk_size
        k_start = i * retrieval_block_size
        k_end = k_start + retrieval_block_size
        retrieval_block_mask[q_start:q_end, k_start:k_end] = 1
    attention_mask = torch.cat([retrieval_block_mask, causal_mask], dim=1)
    return attention_mask


def custom_causal_retrieval_block_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    chunk_size: int,
    retrieval_block_size: int,
    draw_attention_mask: bool = False,
):
    """Implements custom causal retrieval block attention"""
    bsz, nhead, q_len, head_dim = query.shape
    _, kv_nhead, k_len, _ = key.shape

    attention_mask = create_custom_causal_retrieval_block_mask(
        q_len, k_len, chunk_size, retrieval_block_size, query.device
    )

    if draw_attention_mask:
        visualize_attention_mask(attention_mask)

    # Inverse the mask to use in the scaled dot product attention
    attn_mask = attention_mask.float().masked_fill(
        attention_mask.logical_not(), float("-inf")
    )

    # Handle case where kv_nhead != nhead (GQA)
    if kv_nhead != nhead:
        if nhead % kv_nhead != 0:
            raise ValueError("nhead must be a multiple of kv_nhead for GQA.")
        repeat_factor = nhead // kv_nhead
        # Repeat key and value in block order ([k0, k0, k1, k1, ...])
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, is_causal=False, scale=scale
    )


def flash_attention_2(query, key, value, scaling) -> torch.Tensor:
    """Implements FlashAttention-2"""
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask=None,
        query_length=seq_len,
        is_causal=True,
        dropout=0.0,
        softmax_scale=scaling,
        sliding_window=None,
        softcap=None,
        use_top_left_mask=False,
        target_dtype=torch.float16,
        position_ids=torch.arange(seq_len, device=query.device),
    )
    return attn_output.transpose(1, 2)


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
    assert torch.allclose(
        output1, output2, atol=TOLERANCE
    ), f"Compiled and non-compiled {prefix} are not close to each other"

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
    return output2


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


def visualize_attention_mask(attention_mask: torch.Tensor) -> None:
    """Visualize the attention mask"""
    # Convert to numpy for visualization
    mask_np = attention_mask.cpu().numpy()

    # Plot the attention mask
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_np, cmap="gray_r", aspect="auto")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title("Custom Causal Retrieval Block Attention Mask")
    plt.colorbar(label="Mask Value")

    # Save the image
    save_path = "attention_mask.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


def compare_causal_attentions(
    query,
    key,
    value,
    scale,
    enable_gqa,
    causal_mask,
    causal_retrieval_block_mask_wo_retrieval,
):
    """Compare different implementations of the causal attentions"""

    # 1. Custom causal attention
    custom_causal_attention_output = compare_speed_with_and_without_torch_compile(
        "Custom causal attention",
        custom_causal_attention,
        query,
        key,
        value,
        scale=scale,
    )

    # 2. Torch causal attention
    torch_causal_attention_output = compare_speed_with_and_without_torch_compile(
        "Torch causal attention",
        torch_causal_attention,
        query,
        key,
        value,
    )

    # 3. Flex attention with causal block mask
    flex_attention_with_causal_block_mask_output = (
        compare_speed_with_and_without_torch_compile(
            "Flex attention with causal block mask",
            flex_attention,
            query,
            key,
            value,
            block_mask=causal_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=False,
        )
    )

    # 4. Flex attention with causal retrieval block mask without retrieval
    flex_attention_with_causal_retrieval_block_mask_wo_retrieval_output = (
        compare_speed_with_and_without_torch_compile(
            "Flex attention with causal retrieval block mask without retrieval",
            flex_attention,
            query,
            key,
            value,
            block_mask=causal_retrieval_block_mask_wo_retrieval,
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=False,
        )
    )

    # 5. Flash attention-2
    if torch.cuda.get_device_capability() >= (8, 0):
        flash_attention_2_output = compare_speed_with_and_without_torch_compile(
            "Flash attention-2",
            flash_attention_2,
            query,
            key,
            value,
            scaling=scale,
        )

    # Compare all the outputs
    assert torch.allclose(
        custom_causal_attention_output, torch_causal_attention_output, atol=TOLERANCE
    ), "Custom causal attention and torch causal attention are not close to each other"
    assert torch.allclose(
        custom_causal_attention_output,
        flex_attention_with_causal_block_mask_output,
        atol=TOLERANCE,
    ), "Custom causal attention and flex attention with causal block mask are not close to each other"
    assert torch.allclose(
        custom_causal_attention_output,
        flex_attention_with_causal_retrieval_block_mask_wo_retrieval_output,
        atol=TOLERANCE,
    ), "Custom causal attention and flex attention with causal retrieval block mask without retrieval are not close to each other"
    # if torch.cuda.get_device_capability() >= (8, 0):
    #     assert torch.allclose(custom_causal_attention_output.to(flash_attention_2_output.dtype), flash_attention_2_output, atol=TOLERANCE), "Custom causal attention and flash attention-2 are not close to each other"

    return None


def compare_causal_retrieval_block_attention(
    query,
    key,
    value,
    scale,
    enable_gqa,
    chunk_size,
    retrieval_block_size,
    causal_retrieval_block_mask,
):
    """Compare different implementations of the causal retrieval block attention"""

    # 1. Custom causal retrieval block attention
    custom_causal_retrieval_block_attention_output = (
        compare_speed_with_and_without_torch_compile(
            "Custom causal retrieval block attention",
            custom_causal_retrieval_block_attention,
            query,
            key,
            value,
            scale=scale,
            chunk_size=chunk_size,
            retrieval_block_size=retrieval_block_size,
        )
    )

    # 2. Flex attention with causal retrieval block mask
    flex_attention_with_causal_retrieval_block_mask_output = (
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

    # 3. Flex attention without block mask
    flex_attention_without_block_mask_output = (
        compare_speed_with_and_without_torch_compile(
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
    )

    assert torch.allclose(
        custom_causal_retrieval_block_attention_output,
        flex_attention_with_causal_retrieval_block_mask_output,
        atol=TOLERANCE,
    ), "Flex attention with causal retrieval block mask and custom causal retrieval block attention are not close to each other"
    assert not torch.allclose(
        custom_causal_retrieval_block_attention_output,
        flex_attention_without_block_mask_output,
        atol=TOLERANCE,
    ), "Custom causal retrieval block attention and flex attention without block mask are close to each other"

    return None


@hydra.main(version_base=None, config_path="/home/user/RALM/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Configs
    # Model architecture parameters
    bsize = 48
    total_dim = cfg.architecture[cfg.model.architecture].hidden_size
    nhead = cfg.architecture[cfg.model.architecture].num_attention_heads
    kv_nhead = cfg.architecture[cfg.model.architecture].num_key_value_heads
    head_dim = total_dim // nhead
    enable_gqa = kv_nhead != nhead  # Group Query Attention

    # Input and scaling parameters
    input_length = cfg.model.max_length
    chunk_size = cfg.model.input_chunk_size
    scale = head_dim**-0.5

    # Retrieval parameters
    retrieval_block_size = (
        cfg.model.retrieval_chunk_size
        * cfg.model.num_chunks_per_group
        * cfg.model.num_groups_per_block
    )

    # Calculate number of retrieval blocks needed
    num_block_per_input = math.ceil(input_length / chunk_size) - 1

    # Calculate total sequence length including retrieval blocks
    retrieval_input_length = retrieval_block_size * num_block_per_input
    kv_with_retrieval_length = input_length + retrieval_input_length
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
    causal_retrieval_block_mask_without_retrieval = get_causal_retrieval_block_mask(
        input_length=input_length,
        retrieval_block_num=0,
        kv_with_retrieval_length=input_length,
        input_chunk_size=chunk_size,
        retrieval_block_size=retrieval_block_size,
        device=device,
    )
    causal_mask = get_causal_block_mask(
        input_length=input_length,
        kv_with_retrieval_length=input_length,
        device=device,
    )

    compare_causal_attentions(
        query,
        key[:, :, -input_length:, :],
        value[:, :, -input_length:, :],
        scale,
        enable_gqa,
        causal_mask,
        causal_retrieval_block_mask_without_retrieval,
    )
    compare_causal_retrieval_block_attention(
        query,
        key,
        value,
        scale,
        enable_gqa,
        chunk_size,
        retrieval_block_size,
        causal_retrieval_block_mask,
    )

    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
