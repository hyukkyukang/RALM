import logging

import hkkang_utils.misc as misc_utils
import hydra
from omegaconf import DictConfig

from src.retrieval import RetrievedChunkDataset

logger = logging.getLogger("CombineRetrievedChunks")


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    assert cfg.target_dataset is not None, "Set the target dataset in the config"
    retrieved_chunk_dataset = RetrievedChunkDataset(dataset_name=cfg.target_dataset, cfg=cfg.retrieval, global_cfg=cfg)
    logger.info(f"Length of retrieved chunk dataset: {len(retrieved_chunk_dataset)}")
    logger.info(f"First 2 chunk ids: {retrieved_chunk_dataset[:2]}")
    logger.info(f"Last 2 chunk ids: {retrieved_chunk_dataset[-2:]}")
    logger.info("Done!")


# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
