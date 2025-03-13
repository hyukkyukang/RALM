"""Generates a document causal attention mask based on a document ID tensor"""

import logging
import math

import hkkang_utils.misc as misc_utils
import hydra
import torch

# Install from https://github.com/pytorch-labs/attention-gym
from attn_gym import visualize_attention_scores
from omegaconf import DictConfig

from src.model.rellama.mask import generate_causal_retrieval_mask_mod


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Visualize the attention scores of document causal mask mod.

    Args:
        device (str): Device to use for computation. Defaults to "cpu".
    """
    bsize = 1
    total_dim = cfg.model.architecture.hidden_size
    nhead = cfg.model.architecture.num_attention_heads
    input_length = cfg.model.max_length
    kv_nhead = cfg.model.architecture.num_key_value_heads
    head_dim = total_dim // nhead
    chunk_size = cfg.model.input_chunk_size
    num_chunks_per_block = cfg.model.retrieval_chunk_num
    retrieval_block_size = chunk_size * num_chunks_per_block
    num_block_per_input = math.ceil(input_length / chunk_size) - 1
    retrieval_block_len = retrieval_block_size * num_block_per_input
    kv_with_retrieval_length = input_length + retrieval_block_len
    print(
        f"input_length: {input_length}, kv_with_retrieval_length: {kv_with_retrieval_length}"
    )

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
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
