from typing import Callable, List, Optional

import torch

from src.utils import is_torch_compile_possible

# Input length must be greater than this value to use torch compile for flex attention
FLEX_ATT_TORCH_COMPILE_MIN_BLOCK_SIZE = 128


def get_document_ids(
    input_length: int,
    input_chunk_size: int,
    retrieval_block_size: int,
    retrieval_block_num: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate document IDs for retrieval-augmented attention.

    This function creates a tensor that maps each position in the sequence to a document ID.
    Positions from the same document (input chunk or retrieval block) share the same ID,
    which allows the attention mechanism to identify which tokens can attend to each other.

    Args:
        input_length: Total length of the input sequence
        input_chunk_size: Size of each input chunk
        retrieval_block_size: Size of each retrieval block
        retrieval_block_num: Number of retrieval blocks
        device: Device to place the tensor on

    Returns:
        A tensor of document IDs for each position in the sequence
    """
    # Calculate where the actual input starts (after all retrieval blocks)
    input_start_idx = retrieval_block_num * retrieval_block_size

    # Initialize the document ID tensor
    # Total length includes both retrieval blocks and input sequence
    all_input_len = input_length + retrieval_block_size * retrieval_block_num
    document_ids = torch.zeros(all_input_len, dtype=torch.int32, device=device)

    # Mark all input positions with -1 to prevent default connections
    # This will be selectively overridden later for specific connections
    document_ids[input_start_idx:] = -1

    # Assign IDs to retrieval blocks (IDs start from 1)
    # Each retrieval block gets a unique ID that matches its corresponding input chunk
    for i in range(retrieval_block_num):
        block_id = i + 1  # Start from 1 to distinguish from the default 0
        start_idx = retrieval_block_size * i
        end_idx = retrieval_block_size * (i + 1)
        document_ids[start_idx:end_idx] = block_id

    # Assign IDs to input chunks (starting from the second chunk)
    # Each input chunk gets the same ID as its corresponding retrieval block
    for i in range(1, retrieval_block_num + 1):
        block_id = i
        start_idx = input_start_idx + input_chunk_size * i

        # Handle the last chunk which might be shorter
        if i == retrieval_block_num:
            end_idx = all_input_len
        else:
            end_idx = input_start_idx + input_chunk_size * (i + 1)

        document_ids[start_idx:end_idx] = block_id

    return document_ids


def generate_causal_retrieval_mask_mod(
    input_length: int,
    retrieval_block_num: int,
    input_chunk_size: int = 64,
    retrieval_block_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> Callable:
    """
    Generate a function that creates a custom attention mask for retrieval-augmented models.

    This mask combines standard causal masking with document-based masking to enable:
    1. Standard causal attention within the input sequence
    2. Attention between input chunks and their corresponding retrieval blocks
    3. Prevention of attention between unrelated chunks/blocks

    Args:
        input_length: Total length of the input sequence
        retrieval_block_num: Number of retrieval blocks
        input_chunk_size: Size of each input chunk (default: 64)
        retrieval_block_size: Size of each retrieval block (default: 128)

    Returns:
        A function that generates the appropriate attention mask
    """
    # Smaller input length will cause error with torch compile
    assert (
        not is_torch_compile_possible()
        or input_length >= FLEX_ATT_TORCH_COMPILE_MIN_BLOCK_SIZE
    ), f"input_length must be greater than or equal to {FLEX_ATT_TORCH_COMPILE_MIN_BLOCK_SIZE}, but got {input_length}"

    # Calculate where the actual input starts (after all retrieval blocks)
    INPUT_START_IDX = retrieval_block_num * retrieval_block_size

    # Generate document IDs that map positions to their document/chunk
    document_ids = get_document_ids(
        input_length=input_length,
        input_chunk_size=input_chunk_size,
        retrieval_block_size=retrieval_block_size,
        retrieval_block_num=retrieval_block_num,
        device=device,
    )

    def causal_retrieval_mask_mod(b, h, q_idx, kv_idx):
        """
        b: batch index
        h: head index
        q_idx: query position index
        kv_idx: key position index

        Generate the attention mask for a specific query-key pair.

        This function is called by PyTorch's attention mechanism for each position.
        It combines two types of masks:
        1. A causal mask (can only attend to previous positions)
        2. A document mask (can only attend to positions in the same document)

        Args:
            b: Batch index (unused but required by PyTorch)
            h: Head index (unused but required by PyTorch)
            q_idx: Query position index
            kv_idx: Key position index

        Returns:
            Boolean tensor indicating whether attention is allowed (True) or blocked (False)
        """
        # Create standard causal mask (can only attend to previous positions)
        # We add INPUT_START_IDX to q_idx because the query is in the input space
        # but needs to be aligned with the full sequence including retrieval blocks
        causal_mask = (q_idx + INPUT_START_IDX) >= kv_idx

        # Block attention from input to retrieval blocks by default
        # This prevents input tokens from attending to retrieval blocks they shouldn't see
        causal_mask = torch.where(
            kv_idx < INPUT_START_IDX,  # If key is in a retrieval block
            False,  # Block attention by default
            causal_mask,  # Otherwise keep the causal mask
        )

        # Create document-based mask (can only attend to same document)
        # This allows input chunks to attend to their corresponding retrieval blocks
        document_mask = document_ids[q_idx + INPUT_START_IDX] == document_ids[kv_idx]

        # Block document-based attention within the input sequence
        # This ensures we only use document mask for retrieval blocks
        document_mask = torch.where(
            kv_idx >= INPUT_START_IDX,  # If key is in the input sequence
            False,  # Block document-based attention
            document_mask,  # Otherwise keep the document mask
        )

        # Combine masks with OR operation
        # A position can attend if either:
        # 1. It's allowed by causal masking (within input sequence)
        # 2. It's allowed by document masking (between input chunks and their retrieval blocks)
        return causal_mask | document_mask

    return causal_retrieval_mask_mod


def generate_causal_mask_mod(
    query_length: int,
) -> Callable:
    """
    Generate a function that creates a custom attention mask for retrieval-augmented models.

    This mask combines standard causal masking with document-based masking to enable:
    1. Standard causal attention within the input sequence

    Args:
        query_length: Total length of the input sequence

    Returns:
        A function that generates the appropriate attention mask
    """
    # Smaller input length will cause error with torch compile
    assert (
        query_length >= FLEX_ATT_TORCH_COMPILE_MIN_BLOCK_SIZE
    ), f"query_length must be greater than or equal to {FLEX_ATT_TORCH_COMPILE_MIN_BLOCK_SIZE}, but got {query_length}"

    def causal_mask_mod(b, h, q_idx, kv_idx):
        """
        Generate the attention mask for a specific query-key pair.

        This function is called by PyTorch's attention mechanism for each position.
        It combines two types of masks:
        1. A causal mask (can only attend to previous positions)

        Args:
            b: Batch index (unused but required by PyTorch)
            h: Head index (unused but required by PyTorch)
            q_idx: Query position index
            kv_idx: Key position index

        Returns:
            Boolean tensor indicating whether attention is allowed (True) or blocked (False)
        """
        # Create standard causal mask (can only attend to previous positions)
        causal_mask = q_idx >= kv_idx

        return causal_mask

    return causal_mask_mod
