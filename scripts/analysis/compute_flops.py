import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import torch


def silent_warn_once(*args, **kwargs):
    pass


torch._dynamo.utils.warn_once = silent_warn_once
import logging
from typing import *

import hkkang_utils.misc as misc_utils
import hydra
from omegaconf import DictConfig

from src.dataset import DataModule
from src.model import LightningModule
from src.model.utils import calculate_FLOPs

logger = logging.getLogger("FLOPsCounter")


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Initialize lightning module and call prepare_data to figure out the length of the dataset
    data_module = DataModule(cfg=cfg)
    data_module.prepare_data()

    # Initialize lightning module
    lightning_module = LightningModule(
        cfg=cfg,
        total_optimization_steps=0,
        tokenizer=data_module.tokenizer,
    )

    # Create random input
    max_seq_len = 1024

    flops = calculate_FLOPs(
        model=lightning_module.model,
        tokenizer=data_module.tokenizer,
        max_seq_len=max_seq_len,
    )
    print(f"FLOPs: {flops}")
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
