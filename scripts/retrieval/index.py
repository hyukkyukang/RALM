import logging

import hkkang_utils.misc as misc_utils
import hydra
from omegaconf import DictConfig

from src.retrieval import SentenceTransformerCorpusIndexer

logger = logging.getLogger("Indexing")


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    indexer = SentenceTransformerCorpusIndexer(
        cfg=cfg.retrieval.indexing, global_cfg=cfg
    )
    indexer()
    logger.info("Indexing complete.")


# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
