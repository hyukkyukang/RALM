import logging
import math
from typing import *

import hkkang_utils.misc as misc_utils
import hkkang_utils.time as time_utils
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

logger = logging.getLogger("ATTENTION")
torch._dynamo.config.cache_size_limit = 1000
torch.set_float32_matmul_precision("high")

NUM_RUNS = 1000  # Number of times each test runs

def get_document_ids(
    input_length: int,
    input_chunk_size: int,
    retrieval_block_size: int,
    retrieval_block_num: int,
    device: torch.device,
) -> torch.Tensor:
    input_start_idx = retrieval_block_num * retrieval_block_size

    # Initialize the document id
    all_input_len = input_length + retrieval_block_size * retrieval_block_num
    document_ids = torch.zeros(all_input_len, dtype=torch.int32, device=device)

    # Remove connection between retrieval chunk and input chunk, by setting the input chunk id to be -1
    document_ids[input_start_idx:] = -1

    # Set the id for each retrieval blocks (begin from id 1, from the first retrieval chunk)
    for i in range(retrieval_block_num):
        block_id = i + 1
        start_idx = retrieval_block_size * i
        end_idx = retrieval_block_size * (i + 1)
        document_ids[start_idx:end_idx] = block_id

    # Set the id for each input chunks (begin from id 1, from the second input chunk)
    for i in range(1, retrieval_block_num + 1):
        block_id = i
        start_idx = input_start_idx + input_chunk_size * i
        # Handle case where there are less retrieval blocks
        if i == retrieval_block_num:
            end_idx = all_input_len
        else:
            end_idx = input_start_idx + input_chunk_size * (i + 1)
        document_ids[start_idx:end_idx] = block_id

    return document_ids


def get_causal_retrieval_block_mask(
    input_length: int,
    retrieval_block_num: int,
    kv_with_retrieval_length: Optional[int] = None,
    input_chunk_size: int = 64,
    retrieval_block_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> Callable:
    # Configs
    Q_LEN = input_length
    KV_LEN = kv_with_retrieval_length
    INPUT_START_IDX = retrieval_block_num * retrieval_block_size

    document_ids = get_document_ids(
        input_length=input_length,
        input_chunk_size=input_chunk_size,
        retrieval_block_size=retrieval_block_size,
        retrieval_block_num=retrieval_block_num,
        device=device,
    )

    def causal_retrieval_mask_mod(b, h, q_idx, kv_idx):
        # Basic causal mask
        causal_mask = (q_idx + INPUT_START_IDX) >= kv_idx

        # Additional document mask
        document_mask = document_ids[q_idx + INPUT_START_IDX] == document_ids[kv_idx]

        # Remove the input chunk to retrieval chunk connection
        causal_mask = torch.where(
            kv_idx <= INPUT_START_IDX,
            torch.zeros_like(causal_mask, dtype=torch.bool),
            causal_mask,
        )

        # Remove input chunk to input chunk connection (should only handle in causal mask)
        document_mask = torch.where(
            kv_idx >= INPUT_START_IDX,
            torch.zeros_like(document_mask, dtype=torch.bool),
            document_mask,
        )
        
        return causal_mask | document_mask

    block_mask = create_block_mask(
        causal_retrieval_mask_mod,
        1,
        1,
        Q_LEN,
        KV_LEN,
        device=document_ids.device,
    )

    return block_mask


def torch_causal_attention(query_states, key_states, value_states):
    """
    Computes standard causal attention using scaled dot-product attention.
    Ensures key/value heads match query heads using Grouped Query Attention (GQA).
    """
    bsz, nhead, q_len, head_dim = query_states.shape
    _, kv_nhead, k_len, _ = key_states.shape

    # Handle case where kv_nhead != nhead
    if kv_nhead != nhead:
        if nhead % kv_nhead != 0:
            raise ValueError("nhead must be a multiple of kv_nhead for GQA.")

        repeat_factor = nhead // kv_nhead
        key_states = key_states.repeat(1, repeat_factor, 1, 1)
        value_states = value_states.repeat(1, repeat_factor, 1, 1)

    return F.scaled_dot_product_attention(
        query_states, key_states, value_states, attn_mask=None, is_causal=True
    )


def custom_causal_attention(query, key, value, scale):
    """Implements standard causal attention while handling different sequence lengths and head mismatch."""
    bsz, nhead, q_len, head_dim = query.shape
    _, kv_nhead, k_len, _ = key.shape

    if kv_nhead != nhead:
        if nhead % kv_nhead != 0:
            raise ValueError(
                "Number of heads in query must be a multiple of heads in key/value for GQA."
            )

        repeat_factor = nhead // kv_nhead
        key = key.repeat(1, repeat_factor, 1, 1)
        value = value.repeat(1, repeat_factor, 1, 1)

    # Compute scaled dot product attention
    qk = torch.einsum("bhqd, bhkd -> bhqk", query, key) * scale

    # Create causal mask dynamically
    causal_mask = torch.tril(torch.ones((q_len, k_len), device=qk.device)).bool()

    # Apply mask
    qk.masked_fill_(~causal_mask, float("-inf"))
    attn_weights = F.softmax(qk, dim=-1)

    return torch.einsum("bhqk, bhkd -> bhqd", attn_weights, value)



def compare_speed_with_and_without_torch_compile(prefix: str, f1, *args, **kwargs):
    """Run benchmark 100 times and get the average execution time for both compiled and non-compiled functions."""
    f2 = torch.compile(f1)

    # Warm-up
    f1(*args, **kwargs)
    f2(*args, **kwargs)

    # Measure without compilation
    time_no_compile = []
    for _ in range(NUM_RUNS):
        t = time_utils.Timer(f"{prefix} without torch compile")
        with t.measure(False):
            f1(*args, **kwargs)
        time_no_compile.append(t.elapsed_time)

    # Measure with compilation
    time_with_compile = []
    for _ in range(NUM_RUNS):
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


def main():
    # Configs
    bsize = 48
    total_dim = 768
    nhead = 12
    kv_nhead = 3
    head_dim = total_dim // nhead
    scale = head_dim ** -0.5
    input_length = 128
    chunk_size = 64
    num_chunks_per_block = 2
    retrieval_block_size = chunk_size * num_chunks_per_block
    num_block_per_input = math.ceil(input_length / chunk_size) -1
    retrieval_block_len = retrieval_block_size * num_block_per_input
    kv_with_retrieval_length = input_length + retrieval_block_len
    print(f"input_length: {input_length}, kv_with_retrieval_length: {kv_with_retrieval_length}")

    # Device
    device = torch.device("cuda")

    # Get block mask
    block_mask = get_causal_retrieval_block_mask(
        input_length=input_length,
        retrieval_block_num=num_block_per_input,
        kv_with_retrieval_length=kv_with_retrieval_length,
        input_chunk_size=chunk_size,
        retrieval_block_size=retrieval_block_size,
        device=device,
    )

    # Compare speeds with and without torch.compile
    compare_speed_with_and_without_torch_compile(
        "Flex attention without block mask",
        flex_attention,
        torch.randn(bsize, nhead, input_length, head_dim, device=device, requires_grad=True),
        torch.randn(bsize, kv_nhead, kv_with_retrieval_length, head_dim, device=device, requires_grad=True),
        torch.randn(bsize, kv_nhead, kv_with_retrieval_length, head_dim, device=device, requires_grad=True),
        block_mask=None,
        scale=scale,
        enable_gqa=True,
        return_lse=False,
    )

    # compare_speed_with_and_without_torch_compile(
    #     "Flex attention with block mask",
    #     flex_attention,
    #     torch.randn(bsize, nhead, input_length, head_dim, device=device, requires_grad=True),
    #     torch.randn(bsize, kv_nhead, kv_with_retrieval_length, head_dim, device=device, requires_grad=True),
    #     torch.randn(bsize, kv_nhead, kv_with_retrieval_length, head_dim, device=device, requires_grad=True),
    #     block_mask=block_mask,
    #     scale=scale,
    #     enable_gqa=True,
    #     return_lse=False,
    # )

    # compare_speed_with_and_without_torch_compile(
    #     "Custom causal attention",
    #     custom_causal_attention,
    #     torch.randn(bsize, nhead, input_length, head_dim, device=device, requires_grad=True),
    #     torch.randn(bsize, kv_nhead, kv_with_retrieval_length, head_dim, device=device, requires_grad=True),
    #     torch.randn(bsize, kv_nhead, kv_with_retrieval_length, head_dim, device=device, requires_grad=True),
    #     scale=scale,
    # )

    # compare_speed_with_and_without_torch_compile(
    #     "Torch causal attention",
    #     torch_causal_attention,
    #     torch.randn(bsize, nhead, input_length, head_dim, device=device, requires_grad=True),
    #     torch.randn(bsize, kv_nhead, kv_with_retrieval_length, head_dim, device=device, requires_grad=True),
    #     torch.randn(bsize, kv_nhead, kv_with_retrieval_length, head_dim, device=device, requires_grad=True),
    # )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()