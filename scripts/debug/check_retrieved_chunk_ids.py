import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import logging
import math
from typing import *

import hkkang_utils.misc as misc_utils
import hydra
import tqdm
from omegaconf import DictConfig

from src.retrieval.chunk_dataset import RetrievedChunkDataset
from src.retrieval.utils import convert_passage_id_to_global_chunk_ids

logger = logging.getLogger("PL_Trainer")

def check_single_item(retrieved_chunk_dataset: RetrievedChunkDataset, target_passage_id: int) -> None:
    global_chunk_ids = convert_passage_id_to_global_chunk_ids(target_passage_id, retrieved_chunk_dataset.num_chunks_per_passage)[0]
    tmp = retrieved_chunk_dataset[global_chunk_ids]
    print(f"Number of retrieved chunk ids: {len(tmp)}")
    print(tmp)


def check_all_items(retrieved_chunk_dataset: RetrievedChunkDataset) -> None:
    bad_global_chunk_ids = []
    for idx, data in enumerate(tqdm.tqdm(retrieved_chunk_dataset)):
        if len(data) < retrieved_chunk_dataset.num_chunks_per_passage-1:
            bad_global_chunk_ids.append(idx)
    logger.info(f"Number of bad global chunk ids: {len(bad_global_chunk_ids)}")
    logger.info(f"Bad global chunk ids: {bad_global_chunk_ids}")
    return bad_global_chunk_ids

@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    retrieved_chunk_dataset = RetrievedChunkDataset(
        dataset_name="pints_ai",
        cfg=cfg.retrieval,
        global_cfg=cfg,
    )

    # target_passage_id = 4_366_900
    # check_single_item(retrieved_chunk_dataset, target_passage_id)

    check_all_items(retrieved_chunk_dataset)

    logger.info("Done!")



if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
