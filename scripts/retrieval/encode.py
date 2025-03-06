import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
from typing import *

import hkkang_utils.misc as misc_utils
import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig

from src.dataset import DataModule
from src.retrieval import SentenceTransformerEncoder
from src.utils import get_ip

logger = logging.getLogger("Encoding")


def get_partition_indices(rank: int, total_len: int, worker_start_idx_of_current_server: int, total_num_workers: int) -> Tuple[int, int]:
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

def process_partition(rank: int, world_size: int, cfg: DictConfig) -> None:
    # Get the global total number of workers.
    total_num_workers = cfg.retrieval.encoding.num_total_workers
    worker_start_idx_of_current_server = cfg.retrieval.encoding.worker_start_idx_of_current_server
    num_workers_in_current_server = cfg.retrieval.encoding.num_workers_in_current_server

    assert num_workers_in_current_server == world_size, f"The number of workers in the current server ({num_workers_in_current_server}) must be equal to the number of GPUs ({world_size})."
    
    # Set the device for this process
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Create a sub-directory for this GPU process to avoid file name collisions.
    server_ip: str = get_ip().replace(".", "_")
    save_dir: str = os.path.join(cfg.retrieval.embedding_dir_path, server_ip, f"gpu_{rank}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Instantiate the encoder and move the model to the designated GPU.
    encoder = SentenceTransformerEncoder(
        model_name=cfg.retrieval.encoding.model_name,
        src_tokenizer_name=cfg.model.base_name,
        save_dir_path=save_dir,
        device=device,
        enable_torch_compile=cfg.use_torch_compile,
        chunk_size=cfg.retrieval.encoding.chunk_size,
        passage_size=cfg.retrieval.encoding.passage_size
    )

    # Load the dataset from disk.
    data_module = DataModule(cfg=cfg)
    data_module.setup()
    dataset = data_module.train_dataset
    total_len = len(dataset)

    # Partition the dataset indices.
    print("total_len: ", total_len)
    start_idx, end_idx = get_partition_indices(rank, total_len, worker_start_idx_of_current_server, total_num_workers)

    logger.info(f"GPU {rank}: processing indices [{start_idx}:{end_idx}] of {total_len}")

    # Encode the partition.
    encoder.encode_dataset(
        dataset,
        batch_size=cfg.retrieval.encoding.batch_size,
        dataset_start_idx=start_idx,
        dataset_end_idx=end_idx,
        save_in_disk=True,
        num_dataloader_workers=cfg.retrieval.encoding.num_dataloader_workers,
    )

@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Determine the number of GPUs available.
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info(f"Starting encoding with {world_size} GPU(s).")
    
    if world_size > 1:
        # Spawn a process per GPU.
        mp.spawn(process_partition, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        process_partition(0, world_size, cfg)

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()