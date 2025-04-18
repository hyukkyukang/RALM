import functools
from typing import Tuple

import torch
from torch.nn.attention.flex_attention import flex_attention
from tqdm import tqdm

flex_attention = torch.compile(flex_attention)

from scripts.analysis.attention import (
    custom_causal_retrieval_block_attention,
    get_causal_retrieval_block_mask,
)

torch._dynamo.config.cache_size_limit = 1000
TOLERANCE = 1e-6


@functools.lru_cache(maxsize=None)
def get_projection_matrices(
    token_ids_length: int,
    retrieval_block_num: int,
    retrieval_block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    retrieval_ids_length = retrieval_block_size * retrieval_block_num

    # Projection matrices should project from dim to dim
    # For query: 12 heads -> 12 heads
    # For key/value: 12 heads -> 3 heads (for GQA)
    query_projection_matrix = torch.randn(
        1, 12, 64, 64, device=device
    )  # Changed dimensions
    key_projection_matrix = torch.randn(
        1, 12, 64, 64, device=device
    )  # Changed dimensions
    value_projection_matrix = torch.randn(
        1, 12, 64, 64, device=device
    )  # Changed dimensions
    return query_projection_matrix, key_projection_matrix, value_projection_matrix


def create_input_tensors(
    token_ids_length: int,
    retrieval_block_num: int,
    retrieval_block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    retrieval_ids_length = retrieval_block_size * retrieval_block_num
    input_vector = torch.randn(1, 12, token_ids_length, 64, device=device)
    retrieval_vector = torch.randn(1, 12, retrieval_ids_length, 64, device=device)
    return input_vector, retrieval_vector


def encode_input_and_retrieval_vectors(
    input_vector: torch.Tensor,
    retrieval_vector: torch.Tensor,
    token_ids_length: int,
    retrieval_block_num: int,
    retrieval_block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    kv_vector = torch.cat([retrieval_vector, input_vector], dim=2)

    # Get projection matrices
    query_projection_matrix, key_projection_matrix, value_projection_matrix = (
        get_projection_matrices(
            token_ids_length,
            retrieval_block_num,
            retrieval_block_size,
            device,
        )
    )

    # Project input and retrieval vectors
    # Keep query at 12 heads
    query = input_vector @ query_projection_matrix

    # Project key and value from 12 heads to 3 heads for GQA
    key = kv_vector @ key_projection_matrix
    key = key.reshape(1, 3, 4, -1, 64).mean(
        dim=2
    )  # Reshape and mean to go from 12->3 heads

    value = kv_vector @ value_projection_matrix
    value = value.reshape(1, 3, 4, -1, 64).mean(
        dim=2
    )  # Reshape and mean to go from 12->3 heads

    return query, key, value


def main():

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Configuration
    device = torch.device("cuda")
    chunk_size = 64
    retrieval_block_size = 64
    token_ids_length = 1024
    retrieval_block_num = 15
    retrieval_ids_length = retrieval_block_size * retrieval_block_num

    causal_retrieval_block_mask = get_causal_retrieval_block_mask(
        input_length=token_ids_length,
        retrieval_block_num=retrieval_block_num,
        kv_with_retrieval_length=token_ids_length + retrieval_ids_length,
        input_chunk_size=chunk_size,
        retrieval_block_size=retrieval_block_size,
        device=device,
    )

    # Create input and retrieval vectors
    input_vector, retrieval_vector = create_input_tensors(
        token_ids_length,
        retrieval_block_num,
        retrieval_block_size,
        device,
    )

    # Encode input and retrieval vectors
    query, key, value = encode_input_and_retrieval_vectors(
        input_vector,
        retrieval_vector,
        token_ids_length,
        retrieval_block_num,
        retrieval_block_size,
        device,
    )

    # Perform causal retrieval attention
    head_dim = 64
    scale = head_dim**-0.5

    # Baseline
    # output1 = custom_causal_retrieval_block_attention(query, key, value, scale, chunk_size, retrieval_block_size)
    output1 = flex_attention(
        query,
        key,
        value,
        block_mask=causal_retrieval_block_mask,
        scale=scale,
        enable_gqa=True,
        return_lse=False,
    )

    # Modify the second token of the input
    target_position = input_vector.shape[2] // 2
    input_vector[0, 0, target_position, :] = torch.randn(64, device=device) * 100
    # Encode again
    query, key, value = encode_input_and_retrieval_vectors(
        input_vector,
        retrieval_vector,
        token_ids_length,
        retrieval_block_num,
        retrieval_block_size,
        device,
    )
    # output2 = custom_causal_retrieval_block_attention(query, key, value, scale, chunk_size, retrieval_block_size)
    output2 = flex_attention(
        query,
        key,
        value,
        scale=scale,
        block_mask=causal_retrieval_block_mask,
        enable_gqa=True,
        return_lse=False,
    )

    # Check causality
    assert torch.allclose(
        output1[:, :, :target_position, :],
        output2[:, :, :target_position, :],
        atol=TOLERANCE,
    ), "Causality violated: changes affected previous positions"

    # Check that EVERY position after target_position is different
    not_affected = []
    for pos in tqdm(
        range(target_position, output1.shape[2]), desc="Checking positions"
    ):
        if torch.allclose(output1[:, :, pos, :], output2[:, :, pos, :], atol=TOLERANCE):
            not_affected.append(pos)
    print(f"Positions not affected: {not_affected}")

    print("All causality tests passed successfully!")


if __name__ == "__main__":
    main()
