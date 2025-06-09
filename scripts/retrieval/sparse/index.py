import hkkang_utils.misc as misc_utils

misc_utils.load_dotenv()
import logging
import time
from multiprocessing import Pool
from typing import *

import hydra
import tqdm
from datasets import load_from_disk
from omegaconf import DictConfig

from src.retrieval.sparse.bm25 import BM25Retriever

logger = logging.getLogger("Index")

SHARD_SCALE = 0.05
NUM_PROC = 80
CACHE_DIR = "/home/user/RALM/data/huggingface/pile"


def materialize_split(split) -> List[List[int]]:
    return split["input_ids"]


def load_tokenized_and_chunked_dataset(
    dir_path: str, shard_id: int, shard_scale: float
) -> List[List[int]]:
    print("Loading dataset...")
    dataset = load_from_disk(dir_path)

    print(f"Scaling down to {shard_scale}")
    # Find the start and end indices of the shard
    start_idx = int(len(dataset) * shard_scale * shard_id)
    end_idx = int(len(dataset) * shard_scale * (shard_id + 1))
    dataset = dataset.select(range(start_idx, end_idx))

    print("Sharding dataset...")
    splits = [
        dataset.shard(num_shards=NUM_PROC, index=i, contiguous=True)
        for i in range(NUM_PROC)
    ]

    print("Materializing shards...")
    start = time.time()
    # Load all chunks in parallel
    with Pool(processes=NUM_PROC) as p:
        results: List[List[List[int]]] = list(
            tqdm.tqdm(p.imap(materialize_split, splits), total=NUM_PROC)
        )
    time_end = time.time()
    print(f"Materialization done in {time_end - start:.2f} seconds")

    # Flatten the results
    return [item for sublist in results for item in sublist]


@hydra.main(
    version_base=None, config_path="/home/user/RALM/config", config_name="config"
)
def main(cfg: DictConfig) -> None:
    # Get the shard ID
    shard_id = 0

    # Get the total number of shards
    total_shards: int = cfg.retrieval.model.indexing.total_shards
    shard_scale = 1.0 / total_shards

    retriever = BM25Retriever(cfg.retrieval, cfg, device="cuda")

    # Load tokenized dataset
    corpus_name = cfg.retrieval.corpus_name
    logger.info(f"Loading the collection for {corpus_name}...")
    from src.dataset import DATASET_REGISTRY

    dataset = DATASET_REGISTRY[corpus_name](
        cfg.dataset[corpus_name], cfg, retriever.collection_tokenizer
    )
    tokenized_and_chunked_dataset: List[List[int]] = load_tokenized_and_chunked_dataset(
        dataset.post_process_cache_path, shard_id, shard_scale
    )
    # Save the index
    print("Indexing...")
    total_time_start = time.time()
    retriever.create_index(tokenized_and_chunked_dataset)
    total_time_end = time.time()
    print(f"Total time: {total_time_end - total_time_start:.2f} seconds")
    print("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    main()
