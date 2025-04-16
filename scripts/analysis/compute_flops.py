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
from calflops import calculate_flops

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
    batch_size = 1
    seq_len = 1024

    input_shape = (batch_size, seq_len)
    flops, macs, params = calculate_flops(model=lightning_module.model, 
                                        input_shape=input_shape,
                                        transformer_tokenizer=data_module.tokenizer,
                                        include_backPropagation=True)
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
