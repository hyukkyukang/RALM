import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import json
import logging
import multiprocessing
from typing import *

import hkkang_utils.misc as misc_utils
import hydra
import tqdm
from datasets import Dataset
from omegaconf import DictConfig

logger = logging.getLogger("CheckRetrievedChunkIds")


def process_shard(shard_tuple: Tuple[int, str]) -> List[Tuple[int, int]]:
    shard_idx, shard_path = shard_tuple
    dataset = Dataset.load_from_disk(shard_path)
    bad_items = []
    for idx, data in enumerate(tqdm.tqdm(dataset, desc="Checking items")):
        if -1 in data["chunk_ids"]:
            bad_items.append((shard_idx, idx))
    return bad_items

def check_all_items_parallel(path: str) -> None:
    # Load the meta data
    with open(path, "r") as f:
        data = json.load(f)
    # Get the shard paths
    shard_paths = data["chunk_ids_shard_paths"]

    args_list = list(enumerate(shard_paths))
    all_bad_items = []

    # Process the shards in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
         for bad_items in tqdm.tqdm(pool.imap_unordered(process_shard, args_list),
                                     total=len(args_list),
                                     desc="Processing shards"):
             if bad_items:
                 all_bad_items.extend(bad_items)

    # Format the bad items
    bad_cnt = len(all_bad_items)
    tmp: List[Tuple[int, int]] = []
    for shard_idx, data_idx in all_bad_items:
        tmp.append((shard_idx, data_idx))
    
    # Write the bad items to a file
    with open("bad_shard_ids.json", "w") as f:
        json.dump(tmp, f, indent=4)
    
    logger.info(f"Bad shard count: {bad_cnt}")


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    path = "/root/RETRO/data/retrieval/chunk_ids/pints_ai/meta.json"
    check_all_items_parallel(path)
    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()