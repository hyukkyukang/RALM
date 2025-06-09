import os
from multiprocessing import Pool, cpu_count
from typing import List
import time
import tqdm
from datasets import load_from_disk

NUM_PROC = cpu_count()
print(f"Using {NUM_PROC} processes")


def count_tokens_in_range(args) -> int:
    """
    Count tokens in a slice of the dataset without loading the entire dataset into RAM.
    Each process handles a separate range of indices.
    """
    dataset_path, start_idx, end_idx, process_idx = args
    # Wait 1 minute per process id
    time.sleep(60 * process_idx)

    dataset = load_from_disk(dataset_path, keep_in_memory=False)
    total = 0
    print(f"[PID {os.getpid()}] Processing range {start_idx} to {end_idx}")

    # Use select to get only the range we need
    subset = dataset.select(range(start_idx, end_idx))
    for item in tqdm.tqdm(subset, desc=f"Processing range {start_idx} to {end_idx}"):
        total += len(item["input_ids"])
        # Explicitly delete the item to help garbage collection
        del item

    print(f"Total in process {process_idx}: {total}")

    # Clear the subset reference
    del subset
    # Clear the dataset reference
    del dataset

    return total


def count_tokens_multiprocess(dataset_path: str) -> None:
    """
    Count the total number of tokens in a Hugging Face dataset using multiprocessing.
    This avoids full memory loading by splitting index ranges.
    """
    total_len = 177_007_316

    print(f"Total number of samples: {total_len}")
    chunk_size = total_len // NUM_PROC

    index_ranges = [
        (
            dataset_path,
            i * chunk_size,
            (i + 1) * chunk_size if i < NUM_PROC - 1 else total_len,
            i,
        )
        for i in range(NUM_PROC)
    ]

    print(f"Counting tokens using {NUM_PROC} processes...")
    with Pool(processes=NUM_PROC) as p:
        token_counts: List[int] = list(
            tqdm.tqdm(p.imap(count_tokens_in_range, index_ranges), total=NUM_PROC)
        )

    total_tokens = sum(token_counts)
    print(f"\n✅ Total number of tokens: {total_tokens}")


if __name__ == "__main__":
    # Replace this with your dataset path
    dataset_dir_path = os.path.join(
        "/home/user/RALM/data/huggingface/pile", "meta-llama_Llama-3.2-1B_tokenized"
    )
    count_tokens_multiprocess(dataset_dir_path)
