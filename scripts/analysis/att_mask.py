"""Generates a document causal attention mask based on a document ID tensor"""

import math

import torch
from attn_gym import visualize_attention_scores

from src.model.rellama.mask import generate_causal_retrieval_mask_mod


def main():
    """Visualize the attention scores of document causal mask mod.

    Args:
        device (str): Device to use for computation. Defaults to "cpu".
    """
    input_length = 128    # Configs
    bsize = 1
    total_dim = 768
    nhead = 12
    kv_nhead = 3
    head_dim = total_dim // nhead
    input_length = 1024
    chunk_size = 64
    num_chunks_per_block = 2
    retrieval_block_size = chunk_size * num_chunks_per_block
    num_block_per_input = math.ceil(input_length / chunk_size) -1
    retrieval_block_len = retrieval_block_size * num_block_per_input
    kv_with_retrieval_length = input_length + retrieval_block_len
    print(f"input_length: {input_length}, kv_with_retrieval_length: {kv_with_retrieval_length}")

    query = torch.randn(bsize, nhead, input_length, head_dim)
    key = torch.randn(bsize, kv_nhead, kv_with_retrieval_length, head_dim)
    device = torch.device("cpu")

    # Get block mask
    causal_retrieval_mask_mod = generate_causal_retrieval_mask_mod(
        input_length=input_length,
        retrieval_block_num=num_block_per_input,
        input_chunk_size=chunk_size,
        retrieval_block_size=retrieval_block_size,
        device=device,
    )
    
    visualize_attention_scores(
        query,
        key,
        mask_mod=causal_retrieval_mask_mod,
        device=device,
        name="causal_retrieval_mask_mod",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .[viz]")

    CLI(main)
