import logging
import os
from datetime import timedelta

import git
import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from src.dataset import RETRODataModule
from src.model import RETROLightningModule
from src.utils import add_config

logger = logging.getLogger("PL_Trainer")


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Get git hash
    repo = git.Repo(search_parent_directories=True)
    add_config(cfg, "git_hash", repo.head.object.hexsha)
    logger.info(f"Git hash: {cfg.git_hash}")
    
    # Set random seed
    logger.info(f"Setting random seed: {cfg._global.seed}")
    L.seed_everything(cfg._global.seed, workers=True)

    # Set configs
    device_cnt = torch.cuda.device_count()
    log_every_n_steps = cfg.training.logging_steps
    val_check_interval = 100 if cfg._global.is_debug else cfg.training.val_check_interval_by_step
    tag_prefix = "debug_" if cfg._global.is_debug else ""
    default_root_dir = os.path.join(cfg._global.root_dir, f"{tag_prefix}{cfg._global.tag}")
    
    # Initialize lightning module
    data_module = RETRODataModule(cfg=cfg)
    # Compute the total number of training steps
    total_steps = (
        (len(data_module.train_dataloader()) // cfg.training.gradient_accumulation_steps)
        * cfg.training.max_epochs
    )
    model = RETROLightningModule(cfg=cfg, total_steps=total_steps) 

    # Trainer initialization with training args
    trainer = L.Trainer(
        deterministic=True,
        max_epochs=cfg.training.max_epochs,
        num_sanity_val_steps=2,
        profiler="simple",
        accelerator="gpu",
        devices=device_cnt,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        default_root_dir=default_root_dir,
        strategy=L.pytorch.strategies.DDPStrategy(
            timeout=timedelta(hours=4), static_graph=True, gradient_as_bucket_view=True
        ),
        callbacks=[
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
            L.pytorch.callbacks.ModelSummary(max_depth=-1),
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=default_root_dir,
                monitor="loss",
                mode="min",
                save_top_k=1,
                save_last=False,
            ),
        ],
    )
    
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
    main()