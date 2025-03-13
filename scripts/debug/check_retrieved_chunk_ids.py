import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import glob
import json
import logging
import os
from datetime import timedelta
from typing import *
import math
import git
import hkkang_utils.misc as misc_utils
import hkkang_utils.slack as slack_utils
import hydra
import lightning as L
import psutil
import torch
import tqdm
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from src.dataset import DataModule
from src.dataset.dataloader import MyProgressBar
from src.model import LightningModule
from src.model.utils import repair_checkpoint
from src.training.checkpoint import TimeBasedCheckpoint
from src.utils import (
    add_config,
    is_main_process,
    is_torch_compile_possible,
    log_if_rank_zero,
    slack_disable_callback,
)
from src.retrieval.chunk_dataset import RetrievedChunkDataset
logger = logging.getLogger("PL_Trainer")


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    retrieved_chunk_dataset = RetrievedChunkDataset(
        dataset_name="pints_ai",
        cfg=cfg.retrieval,
        global_cfg=cfg,
    )
    

    for idx, data in enumerate(tqdm.tqdm(retrieved_chunk_dataset)):
        retrieved_input_ids = data
        retrieval_chunk_num = cfg.model.retrieval_chunk_num
        retrieval_block_size = cfg.model.retrieval_chunk_size * retrieval_chunk_num
        retrieval_block_num = math.ceil(cfg.model.max_length / retrieval_block_size) - 1
        retrieval_chunk_size = cfg.model.input_chunk_size
        assert len(retrieved_input_ids) == retrieval_block_num, f"Retrieved input ids are not of length 15: {len(retrieved_input_ids)} idx: {idx}"
        # Check number of chunks per retrieval block
        assert len(retrieved_input_ids[0]) == retrieval_chunk_num, f"Retrieved input ids are not of length 2: {len(retrieved_input_ids[0])} idx: {idx}"
        # Check length of chunks
        assert len(retrieved_input_ids[0][0]) == retrieval_chunk_size, f"Retrieved input ids are not of length 64: {len(retrieved_input_ids[0][0])} idx: {idx}"
    
    print("Done!")



if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
