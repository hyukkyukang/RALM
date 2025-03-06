import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
import pickle
from typing import *

import hkkang_utils.concurrent as concurrent_utils
import hkkang_utils.misc as misc_utils
import hkkang_utils.slack as slack_utils
import hydra
from datasets import Dataset
from omegaconf import DictConfig

from src.dataset import DataModule
from retrieval.multi_vector_retrieval.xtr_warp import Retriever
from src.utils import slack_disable_callback

logger = logging.getLogger("Retrieval")

def get_global_worker_idx(worker_start_idx_of_current_server: int, current_worker_local_idx: int) -> int:
    # Get the start and end indices for the current worker
    worker_global_idx = worker_start_idx_of_current_server + current_worker_local_idx
    return worker_global_idx

def get_dataset_range_for_current_worker(total_num_items: int, total_num_workers: int, worker_start_idx_of_current_server: int, current_worker_local_idx: int) -> Tuple[int, int]:
    # Split the dataset into num_workers_in_current_server parts
    items_per_process = total_num_items // total_num_workers
    # Get the start and end indices for the current worker
    worker_global_idx = get_global_worker_idx(worker_start_idx_of_current_server, current_worker_local_idx)
    # Get the dataset range for the current worker
    start_idx = worker_global_idx * items_per_process
    end_idx = start_idx + items_per_process
    
    return start_idx, end_idx

def single_retrieval(process_idx: int, data_module: DataModule, global_cfg: DictConfig) -> str:
    # Initialize the retriever
    retriever = Retriever(address=global_cfg.retrieval.address, port=global_cfg.retrieval.port)
    # Get the global process idx
    global_process_idx = get_global_worker_idx(worker_start_idx_of_current_server=global_cfg.retrieval.worker_start_idx_of_current_server, current_worker_local_idx=process_idx)
    # Get the dataset range for the current worker
    start_idx, end_idx = get_dataset_range_for_current_worker(total_num_items=len(data_module), total_num_workers=global_cfg.retrieval.total_num_workers, worker_start_idx_of_current_server=global_cfg.retrieval.worker_start_idx_of_current_server, current_worker_local_idx=process_idx)

    # Get the retrieved data
    retrieved_indices_list: List[List[int]] = []
    for idx in range(start_idx, end_idx):
        # Get the item
        item = data_module[idx]
        # Get the retrieved data
        decoded_text: List[str] = data_module.tokenizer.decode(item["input_ids"], skip_special_tokens=True)
        retrieved_data_indices: List[int] = retriever(decoded_text)
        # Save the retrieved data
        retrieved_indices_list.append(retrieved_data_indices)
    # Save the retrieved indices
    saved_path = os.path.join(data_module.retrieved_data_cache_path, f"retrieved_indices_{global_process_idx}.parquet")
    # Save the retrieved indices
    with open(saved_path, "wb") as f:
        pickle.dump(retrieved_indices_list, f)
    return saved_path


def parallel_retrieval(data_module: DataModule, num_workers: int = 64, global_cfg: DictConfig = None) -> None:
    # Get total number of items
    total_items = len(data_module)
    # Get number of items per process
    items_per_process = total_items // num_workers
    # Initialize multiprocessor
    multiprocessor = concurrent_utils.MultiProcessor(num_workers=num_workers)
    # Run count_tokens for each process
    for i in range(num_workers):
        multiprocessor.run(
            single_retrieval,
            process_idx=i,
            data_module=data_module,
            indices=range(i * items_per_process, (i + 1) * items_per_process),
            global_cfg=global_cfg,
        )
    # Join all processes
    multiprocessor.join()

    # Retrieve the results
    saved_paths: List[str] = multiprocessor.results

    # Combine the results
    retrieved_indices_list: List[List[int]] = []
    for saved_path in saved_paths:
        # Load the dataset
        dataset = data_module.load_from_disk(saved_path)
        retrieved_indices_list.extend(dataset)

    # Save the retrieved indices
    retrieval_dataset: Dataset = Dataset.from_list(retrieved_indices_list)
    retrieval_dataset.save_to_disk(data_module.retrieved_data_cache_path)

    # Remove the temporary datasets
    for saved_path in saved_paths:
        os.remove(saved_path)

    return None


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Initialize lightning module and call prepare_data to figure out the length of the dataset
    data_module = DataModule(cfg=cfg)
    data_module.prepare_data()

    # Conduct retrieval for training data
    logger.info(f"Conducting retrieval for training data {data_module.train_dataset.name}...")
    parallel_retrieval(data_module.train_dataset)

    # Conduct retrieval for validation data
    for dataset in data_module.val_datasets:
        logger.info(f"Conducting retrieval for validation data {dataset.name}...")
        parallel_retrieval(dataset, global_cfg=cfg)

    logger.info(f"Retrieval complete. Total dataset size: {len(data_module)}")

    return None

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    channel = os.getenv("SLACK_CHANNEL_NAME")
    slack_user_id = os.getenv("SLACK_USER_ID")
    assert channel is not None and slack_user_id is not None, "Set the SLACK_CHANNEL_NAME and SLACK_USER_ID in the .env file"
    with slack_utils.notification(
        channel=channel,
        success_msg=f"<@{slack_user_id}> Retrieval complete.",
        error_msg=f"<@{slack_user_id}> Retrieval failed.",
        replies=[],
        disable=False,
        disable_callback=slack_disable_callback,
    ):
        main()
