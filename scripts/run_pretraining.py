import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
from datetime import timedelta

import git
import hkkang_utils.misc as misc_utils
import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig

from src.dataset import ReLLamaDataModule
from src.model import ReLLamaLightningModule
from src.utils import add_config, log_if_rank_zero

logger = logging.getLogger("PL_Trainer")


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    tag_prefix = "debug_" if cfg._global.is_debug else ""
    default_root_dir = os.path.join(
        cfg._global.root_dir_path,
        cfg._global.log_dir,
        f"{tag_prefix}{cfg._global.tag}",
    )

    # Get git hash
    repo = git.Repo(search_parent_directories=True)
    add_config(cfg, "git_hash", repo.head.object.hexsha)
    log_if_rank_zero(logger, f"Git hash: {cfg.git_hash}")

    # Set random seed
    log_if_rank_zero(logger, f"Setting random seed: {cfg._global.seed}")
    L.seed_everything(cfg._global.seed, workers=True)

    # Initialize lightning module and call prepare_data to figure out the length of the dataset
    data_module = ReLLamaDataModule(cfg=cfg)
    data_module.prepare_data()
    # Compute the total number of training steps
    total_steps = (
        len(data_module) // cfg.training.gradient_accumulation_steps
    ) * cfg.training.max_epochs
    model = ReLLamaLightningModule(
        cfg=cfg, total_steps=total_steps, tokenizer=data_module.tokenizer
    )

    # Trainer initialization with training args
    trainer = L.Trainer(
        deterministic=True,
        max_epochs=cfg.training.max_epochs,
        num_sanity_val_steps=0,
        profiler="simple",
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=cfg.training.logging_steps,
        default_root_dir=default_root_dir,
        strategy=L.pytorch.strategies.DDPStrategy(
            timeout=timedelta(hours=4), static_graph=True, gradient_as_bucket_view=True
        ),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            # ModelSummary(max_depth=-1), # Turn this on if you want to see the model architecture (i.e., the parameter names)
            ModelCheckpoint(
                dirpath=default_root_dir,
                monitor="loss",
                mode="min",
                save_top_k=1,
                every_n_train_steps=cfg.training.checkpoint_save_steps,
                save_last=True,
            ),
        ],
    )
    # Prevent logging `hp_metric`
    if trainer.logger:
        trainer.logger.log_hyperparams = lambda params, metrics=None: None

    # Start training
    log_if_rank_zero(logger, "Starting lightning fit...")
    if cfg.training.resume_ckpt_path:
        log_if_rank_zero(
            logger, f"Resuming from checkpoint: {cfg.training.resume_ckpt_path}"
        )
    trainer.fit(model, datamodule=data_module, ckpt_path=cfg.training.resume_ckpt_path)

    log_if_rank_zero(logger, "Training completed successfully!")

    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
