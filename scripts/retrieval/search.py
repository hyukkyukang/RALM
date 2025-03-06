import logging
from typing import *

import hkkang_utils.misc as misc_utils
import hydra
from omegaconf import DictConfig

from src.retrieval import SentenceTransformerRetriever

logger = logging.getLogger("Indexing")

@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    retriever = SentenceTransformerRetriever(cfg=cfg.retrieval, global_cfg=cfg)
    results: List[Union[str, int]] = retriever.search_batch(queries=["What is the capital of France?", "What is the capital of Germany?"], k=5, return_as_text=True)
    print(results)
    print("Done!")

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
