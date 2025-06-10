import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("RetrievalUtils")


def validate_saved_numpy_files(
    paths: List[str], read_file_content: bool = False
) -> bool:
    """
    Validate that all saved embedding files are present and contain embeddings.
    We use the file name to check that all consecutive embeddings are present.

    For instance, a file named "embeddings_16508182_200.npy" indicates that the file contains
    the 16508182nd to 16508381st passage embeddings.

    Parameters:
        paths (List[str]): List of file paths to validate.
        read_file_content (bool): Whether to read the file content to check the number of embeddings.

    Returns:
        bool: True if the files are valid, False otherwise.
    """
    # Parse file names to extract (start_index, count) for each file.
    file_info: List[Tuple[int, int, str]] = []
    for path in paths:
        stem = Path(path).stem  # e.g. "embeddings_16508182_200"
        parts = stem.split("_")
        if len(parts) < 3:
            logger.error(f"Invalid file name format: {path}")
            continue
        try:
            start_index = int(parts[-2])
            count = int(parts[-1])
        except ValueError:
            logger.error(f"Could not parse start index or count in file name: {path}")
            continue
        file_info.append((start_index, count, path))

    # Sort the file info by start_index.
    file_info.sort(key=lambda x: x[0])

    # Validate consecutive passage indices.
    expected_start: int = None
    for start_index, count, path in file_info:
        if read_file_content:
            arr = load_embedding_file(path)
            if arr.shape[0] != count:
                raise ValueError(
                    f"File {path} claims {count} embeddings in its name but contains {arr.shape[0]} rows."
                )

        if expected_start is None:
            expected_start = start_index
        else:
            if start_index != expected_start:
                raise ValueError(
                    f"Non-consecutive file sequence: expected start index {expected_start}, "
                    f"but got {start_index} in file {path}."
                )
        expected_start = start_index + count

    logger.info("Validation of saved embedding files is complete.")
    return True


def load_embedding_file(path: str) -> np.ndarray:
    """
    Load an embedding file as a numpy array using memory mapping for efficiency.

    Parameters:
        path (str): The file path to the .npy embedding file.

    Returns:
        np.ndarray: The embedding array. Reshaped to 2D if originally 1D.
    """
    arr: np.ndarray = np.load(path, mmap_mode="r")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def convert_global_chunk_id_to_passage_id_and_local_chunk_range(
    global_chunk_id: int, num_chunks_per_passage: int, chunk_size: int
) -> Tuple[int, int, int]:
    """
    Convert a global chunk ID to the corresponding passage ID and local chunk ID.
    """
    passage_id = convert_global_chunk_id_to_passage_id(
        global_chunk_id, num_chunks_per_passage
    )
    local_chunk_id = convert_global_chunk_id_to_local_chunk_id(
        global_chunk_id, num_chunks_per_passage
    )
    chunk_start_idx = local_chunk_id * chunk_size
    chunk_end_idx = chunk_start_idx + chunk_size
    return passage_id, chunk_start_idx, chunk_end_idx


def convert_global_chunk_id_to_passage_id_and_local_chunk_id(
    global_chunk_id: int, num_chunks_per_passage: int
) -> Tuple[int, int]:
    """
    Convert a global chunk ID to the corresponding passage ID and local chunk ID.
    """
    passage_id = convert_global_chunk_id_to_passage_id(
        global_chunk_id, num_chunks_per_passage
    )
    local_chunk_id = convert_global_chunk_id_to_local_chunk_id(
        global_chunk_id, num_chunks_per_passage
    )
    return passage_id, local_chunk_id


def convert_global_chunk_id_to_passage_id(
    global_chunk_id: int, num_chunks_per_passage: int
) -> int:
    """
    Convert a global chunk ID to the corresponding passage ID.
    """
    return global_chunk_id // num_chunks_per_passage


def convert_global_chunk_id_to_local_chunk_id(
    global_chunk_id: int, num_chunks_per_passage: int
) -> int:
    """
    Convert a global chunk ID to the corresponding local chunk ID.
    """
    return global_chunk_id % num_chunks_per_passage


def convert_passage_id_to_global_chunk_ids(
    passage_id: int, num_chunks_per_passage: int
) -> List[int]:
    """
    Convert a passage ID to the corresponding global chunk IDs.
    """
    return [
        passage_id * num_chunks_per_passage + i for i in range(num_chunks_per_passage)
    ]
