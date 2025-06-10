import logging
import os
from typing import *

import hkkang_utils.misc as misc_utils
import hkkang_utils.time as time_utils
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.attention.flex_attention import flex_attention

from scripts.analysis.attention import get_causal_retrieval_block_mask

logger = logging.getLogger("ATTENTION")
torch.set_float32_matmul_precision("high")


def run_flex_attention(
    query_states, key_states, value_states, block_mask, scale
) -> float:
    output = flex_attention(
        query_states,
        key_states,
        value_states.float(),
        block_mask=block_mask,
        scale=scale,
        enable_gqa=True,
        return_lse=False,
    )
    timer = time_utils.Timer("Flex Attention (Uncompiled)")
    with timer.measure(False):
        output = flex_attention(
            query_states,
            key_states,
            value_states.float(),
            block_mask=block_mask,
            scale=scale,
            enable_gqa=True,
            return_lse=False,
        )
    return timer.elapsed_time


compiled_run_flex_attention = torch.compile(run_flex_attention)


def run_compiled_flex_attention(
    query_states, key_states, value_states, block_mask, scale
) -> float:
    timer = time_utils.Timer("Flex Attention (Compiled)")
    # compiled_run_flex_attention = torch.compile(run_flex_attention)
    # Warm-up
    compiled_run_flex_attention(
        query_states,
        key_states,
        value_states.float(),
        block_mask=block_mask,
        scale=scale,
    )
    with timer.measure(False):
        output = compiled_run_flex_attention(
            query_states,
            key_states,
            value_states.float(),
            block_mask=block_mask,
            scale=scale,
        )
    return timer.elapsed_time


def setup(rank, world_size):
    """Initializes the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()


def benchmark_attention(rank, world_size):
    """Runs attention benchmarking across multiple GPUs with synchronization."""
    setup(rank, world_size)
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    # Configs
    bsize = 1
    total_dim = 768
    nhead = 12
    kv_nhead = 3
    head_dim = total_dim // nhead
    scale = head_dim**-0.5
    input_length = 86
    chunk_size = 64
    num_chunks_per_block = 1
    retrieval_block_size = chunk_size * num_chunks_per_block
    num_block_per_input = input_length // retrieval_block_size - 1
    retrieval_block_len = retrieval_block_size * num_block_per_input
    kv_with_retrieval_length = input_length + retrieval_block_len

    # Create query, key, value states
    device = torch.device("cuda")
    query_states = torch.randn(bsize, nhead, input_length, head_dim, device=device)
    key_states = torch.randn(
        bsize, kv_nhead, kv_with_retrieval_length, head_dim, device=device
    )
    value_states = torch.randn(
        bsize, kv_nhead, kv_with_retrieval_length, head_dim, device=device
    )

    # Get block mask
    block_mask = get_causal_retrieval_block_mask(
        input_length=input_length,
        retrieval_block_num=num_block_per_input,
        kv_with_retrieval_length=kv_with_retrieval_length,
        input_chunk_size=chunk_size,
        retrieval_block_size=retrieval_block_size,
        device=query_states.device,
    )
    # Run benchmarking
    torch.cuda.synchronize()  # Ensure all devices start at the same time
    uncompiled_elapsed_time = run_flex_attention(
        query_states, key_states, value_states, block_mask=block_mask, scale=scale
    )
    compiled_elapsed_time = run_compiled_flex_attention(
        query_states, key_states, value_states, block_mask=block_mask, scale=scale
    )

    # Create tensors
    uncompiled_tensor = torch.tensor(uncompiled_elapsed_time, device=device)
    compiled_tensor = torch.tensor(compiled_elapsed_time, device=device)

    # Perform all_reduce operations
    dist.all_reduce(uncompiled_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(compiled_tensor, op=dist.ReduceOp.MAX)

    # Store the results back into the variables
    uncompiled_elapsed_time = uncompiled_tensor.item()
    compiled_elapsed_time = compiled_tensor.item()

    # Now log the synchronized values
    if rank == 0:
        logger.info(
            f"[GPU {rank}] Uncompiled Elapsed Time: {uncompiled_elapsed_time:.6f} sec"
        )
        logger.info(
            f"[GPU {rank}] Compiled Elapsed Time: {compiled_elapsed_time:.6f} sec"
        )

    cleanup()


def run_ddp(world_size):
    """Launches multiple processes for synchronized benchmarking."""
    mp.spawn(benchmark_attention, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()

    # Get the number of GPUs available
    world_size = torch.cuda.device_count()
    if world_size < 2:
        logger.warning("DDP requires at least 2 GPUs. Running on a single GPU.")
        benchmark_attention(0, 1)
    else:
        run_ddp(world_size)
