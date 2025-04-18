import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
from datetime import timedelta

import hkkang_utils.misc as misc_utils
import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from src.dataset import DataModule
from src.model import LightningModule
from src.utils import log_if_rank_zero

logger = logging.getLogger("Evaluation")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._dynamo.config.cache_size_limit = 10000


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Make use_torch_compile to False
    cfg.use_torch_compile = False
    cfg.training.precision = 32

    # Load trained model
    if cfg.model.name in ["rellama", "llama"]:
        assert cfg.ckpt_path, "Please provide the path to the checkpoint"
        lightning_module = LightningModule.load_from_checkpoint(
            cfg.ckpt_path, map_location="cpu", training=cfg.training
        )
        lightning_module.cfg = cfg
    elif cfg.model.name == "gpt":
        lightning_module = LightningModule(cfg=cfg).to("cpu")
    else:
        raise ValueError(f"Model name {cfg.model.name} not supported")
    lightning_module.eval()

    # Load data module and model
    data_module = DataModule(cfg=cfg, is_test=True)

    # Get the data
    test_dataloader = data_module.test_dataloader()
    for batch in test_dataloader:
        print(batch)
        stop = 1
        break


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
