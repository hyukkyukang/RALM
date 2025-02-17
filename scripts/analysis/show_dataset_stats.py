import logging
import os

import hkkang_utils.misc as misc_utils
import hydra
from omegaconf import DictConfig
from src.dataset import DataModule

logger = logging.getLogger("Evaluation")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Get the dataset
    data_module = DataModule(cfg=cfg)
    data_module.prepare_data()
    print("Begin computing total number of tokens...")
    print(data_module.train_dataset.total_tokens)
    print("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
