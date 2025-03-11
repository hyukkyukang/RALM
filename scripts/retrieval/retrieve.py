import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import copy
import logging
import os
from typing import *

import hkkang_utils.misc as misc_utils
import hkkang_utils.slack as slack_utils
import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig

from src.dataset import DATASET_REGISTRY
from src.retrieval import SentenceTransformerCorpusRetriever
from src.tokenization import ReLlamaTokenizer
from src.utils import get_ip, get_partition_indices, slack_disable_callback

logger = logging.getLogger("Retrieval")

# Changing the precision of the matmul operation to high causes error when torch compile the flex attention
torch._dynamo.config.cache_size_limit = 1000
torch.set_float32_matmul_precision("high")


def process_partition(rank: int, world_size: int, cfg: DictConfig) -> None:
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    # Get the global total number of workers.
    total_num_workers = cfg.retrieval.retrieving.num_total_workers
    worker_start_idx_of_current_server = (
        cfg.retrieval.retrieving.worker_start_idx_of_current_server
    )
    num_workers_in_current_server = (
        cfg.retrieval.retrieving.num_workers_in_current_server
    )

    assert (
        num_workers_in_current_server == world_size
    ), f"The number of workers in the current server ({num_workers_in_current_server}) must be equal to the number of GPUs ({world_size})."

    # Set the device for this process
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Create a sub-directory for this GPU process to avoid file name collisions.
    server_ip: str = get_ip().replace(".", "_")
    save_dir: str = os.path.join(
        cfg.retrieval.dir_path,
        cfg.retrieval.retrieved_chunk_ids_dir,
        cfg.target_dataset,
        server_ip,
        f"gpu_{rank}",
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Instantiate the retriever and move the model to the designated GPU.
    retriever = SentenceTransformerCorpusRetriever(
        cfg.retrieval, cfg, save_dir_path=save_dir, device=device
    )

    # Instantiate the tokenizer.
    tokenizer = ReLlamaTokenizer.from_pretrained(cfg.model.base_name)

    # Create a temporary global config with retrieval disabled.
    # To avoid recursive loading of the entangled datasets: RetrievedChunkDataset and BaseDataset
    global_cfg_tmp = copy.deepcopy(cfg)
    global_cfg_tmp.model.is_use_retrieval = False

    # Load the dataset from disk.
    dataset_name = cfg.target_dataset
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(
        cfg=global_cfg_tmp.dataset[dataset_name],
        global_cfg=global_cfg_tmp,
        tokenizer=tokenizer,
    )
    dataset.post_processed_data = dataset.load_from_disk(
        dataset.post_process_cache_path
    )
    total_len = len(dataset)

    # Partition the dataset indices.
    print("total_len: ", total_len)
    start_idx, end_idx = get_partition_indices(
        rank, total_len, worker_start_idx_of_current_server, total_num_workers
    )

    logger.info(
        f"GPU {rank}: processing indices [{start_idx}:{end_idx}] of {total_len}"
    )
    retriever.retrieve_dataset(
        dataset,
        batch_size=cfg.retrieval.retrieving.batch_size,
        data_span_start_idx=start_idx,
        data_span_end_idx=end_idx,
        num_dataloader_workers=cfg.retrieval.retrieving.num_dataloader_workers,
    )


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Determine the number of GPUs available.
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info(f"Starting retrieval with {world_size} GPU(s).")

    assert cfg.target_dataset is not None, "Set the target dataset in the config"

    if world_size > 1:
        # Spawn a process per GPU.
        mp.spawn(
            process_partition, args=(world_size, cfg), nprocs=world_size, join=True
        )
    else:
        process_partition(0, world_size, cfg)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    channel = os.getenv("SLACK_CHANNEL_NAME")
    slack_user_id = os.getenv("SLACK_USER_ID")
    assert (
        channel is not None and slack_user_id is not None
    ), "Set the SLACK_CHANNEL_NAME and SLACK_USER_ID in the .env file"
    with slack_utils.notification(
        channel=channel,
        success_msg=f"Retrieval complete.",
        error_msg=f"Retrieval failed.",
        user_id_to_mention=slack_user_id,
        replies=[],
        disable=False,
        disable_callback=slack_disable_callback,
    ):
        main()
