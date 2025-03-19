import os
import socket
from typing import *


def is_main_process() -> bool:
    """Check if the current process is the main process."""
    # If LOCAL_RANK is set, use it; otherwise, assume a single-process scenario.
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"]) == 0
    return True


def get_partition_indices(
    rank: int,
    total_len: int,
    worker_start_idx_of_current_server: int,
    total_num_workers: int,
) -> Tuple[int, int]:
    # Calculate the base size each worker should process
    base_size = total_len // total_num_workers

    # Calculate the global worker index
    global_worker_idx = worker_start_idx_of_current_server + rank

    # Calculate start index for this worker
    start_idx = global_worker_idx * base_size

    # Calculate end index for this worker
    # Workers with index < remainder get one extra item
    remainder = total_len % total_num_workers

    # Adjust start_idx if this worker comes after workers that got extra items
    if global_worker_idx < remainder:
        start_idx += global_worker_idx
        end_idx = start_idx + base_size + 1
    else:
        start_idx += remainder
        end_idx = start_idx + base_size

    # Safety check to ensure we don't exceed total_len
    end_idx = min(end_idx, total_len)

    return start_idx, end_idx


def get_global_worker_idx(
    worker_start_idx_of_current_server: int, current_worker_local_idx: int
) -> int:
    # Get the start and end indices for the current worker
    worker_global_idx = worker_start_idx_of_current_server + current_worker_local_idx
    return worker_global_idx


def get_dataset_range_for_current_worker(
    total_num_items: int,
    total_num_workers: int,
    worker_start_idx_of_current_server: int,
    current_worker_local_idx: int,
) -> Tuple[int, int]:
    # Split the dataset into num_workers_in_current_server parts
    items_per_process = total_num_items // total_num_workers
    # Get the start and end indices for the current worker
    worker_global_idx = get_global_worker_idx(
        worker_start_idx_of_current_server, current_worker_local_idx
    )
    # Get the dataset range for the current worker
    start_idx = worker_global_idx * items_per_process
    end_idx = start_idx + items_per_process

    return start_idx, end_idx


def get_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
