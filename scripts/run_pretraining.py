import logging
import os
from datetime import timedelta

import git
import hkkang_utils.misc as misc_utils
import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from src.dataset import RETRODataModule
from src.model import RETROLightningModule
from src.utils import add_config
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)

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
    logger.info(f"Git hash: {cfg.git_hash}")

    # Set random seed
    logger.info(f"Setting random seed: {cfg._global.seed}")
    L.seed_everything(cfg._global.seed, workers=True)

    # Initialize lightning module and call prepare_data to figure out the length of the dataset
    data_module = RETRODataModule(cfg=cfg)
    data_module.prepare_data()
    # Compute the total number of training steps
    total_steps = (
        len(data_module) // cfg.training.gradient_accumulation_steps
    ) * cfg.training.max_epochs
    model = RETROLightningModule(cfg=cfg, total_steps=total_steps)

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
            ModelSummary(max_depth=-1),
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
    logger.info("Starting training...")
    if cfg.training.resume_ckpt_path:
        logger.info(f"Resuming from checkpoint: {cfg.training.resume_ckpt_path}")
    trainer.fit(model, datamodule=data_module, ckpt_path=cfg.training.resume_ckpt_path)

    logger.info("Training completed successfully!")

    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
