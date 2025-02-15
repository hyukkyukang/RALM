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
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig

from src.dataset import DataModule
from src.model import LightningModule
from src.utils import add_config, log_if_rank_zero

logger = logging.getLogger("PL_Trainer")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 64


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    tag_prefix = "debug_" if cfg.is_debug else ""
    default_root_dir = os.path.join(
        cfg.root_dir_path,
        cfg.log_dir,
        f"{tag_prefix}{cfg.tag}",
    )

    # Get git hash
    repo = git.Repo(search_parent_directories=True)
    add_config(cfg, "git_hash", repo.head.object.hexsha)
    log_if_rank_zero(logger, f"Git hash: {cfg.git_hash}")

    # Set random seed
    log_if_rank_zero(logger, f"Setting random seed: {cfg.seed}")
    L.seed_everything(cfg.seed, workers=True)

    # Initialize lightning module and call prepare_data to figure out the length of the dataset
    data_module = DataModule(cfg=cfg)
    data_module.prepare_data()

    # Compute the total number of training steps for the learning rate scheduler
    total_optimization_steps = (
        len(data_module.train_dataset)
        // (
            cfg.training.gradient_accumulation_steps
            * cfg.training.per_device_batch_size
            * torch.cuda.device_count()
        )
        * cfg.training.max_epochs
    )
    total_training_steps = (
        len(data_module.train_dataset)
        // (cfg.training.per_device_batch_size * torch.cuda.device_count())
        * cfg.training.max_epochs
    )
    log_if_rank_zero(logger, f"Total training steps: {total_training_steps}")
    log_if_rank_zero(logger, f"Total optimization steps: {total_optimization_steps}")

    # Initialize lightning module
    lightning_module = LightningModule(
        cfg=cfg,
        total_optimization_steps=total_optimization_steps,
        tokenizer=data_module.tokenizer,
    )

    # Initialize checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=default_root_dir,
        monitor="NTP_wikitext_perplexity",
        mode="min",
        save_top_k=1,
        every_n_train_steps=cfg.training.checkpoint_save_steps,
        save_last=True,
        save_on_train_epoch_end=False,
        enable_version_counter=True,
    )

    # Trainer initialization with training args
    trainer = L.Trainer(
        deterministic=True,
        max_epochs=cfg.training.max_epochs,
        profiler="simple",
        accelerator="gpu",
        num_sanity_val_steps=cfg.validation.num_sanity_val_steps,
        val_check_interval=cfg.validation.val_check_interval,
        check_val_every_n_epoch=cfg.validation.check_val_every_n_epoch,
        devices=torch.cuda.device_count(),
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.logging_steps,
        default_root_dir=default_root_dir,
        logger=TensorBoardLogger(
            save_dir=default_root_dir, name=cfg.tag, default_hp_metric=False
        ),
        strategy=DDPStrategy(
            timeout=timedelta(hours=4), static_graph=True, gradient_as_bucket_view=True
        ),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            # ModelSummary(max_depth=-1), # Turn this on if you want to see the model architecture (i.e., the parameter names),
            checkpoint_callback,
        ],
    )
    # Start training
    log_if_rank_zero(logger, "Starting lightning fit...")
    if cfg.training.resume_ckpt_path:
        log_if_rank_zero(
            logger, f"Resuming from checkpoint: {cfg.training.resume_ckpt_path}"
        )
    trainer.fit(
        lightning_module,
        datamodule=data_module,
        ckpt_path=cfg.training.resume_ckpt_path,
    )

    log_if_rank_zero(logger, "Training completed successfully!")

    # Rename the modules in the checkpoint when using torch compile
    # For the main process with rank 0 only
    if (
        torch.distributed.get_rank() == 0
        and cfg.use_torch_compile
        and torch.cuda.get_device_capability()[0] >= 7
    ):
        last_checkpoint_path = os.path.join(
            default_root_dir, f"version_{trainer.logger.version}", "last.ckpt"
        )
        print(
            "checkpoint_callback.best_model_path:", checkpoint_callback.best_model_path
        )
        if os.path.exists(last_checkpoint_path):
            log_if_rank_zero(
                logger,
                f"Renaming the modules in the checkpoint ({last_checkpoint_path}) for torch compile...",
            )
            # Load the checkpoint
            checkpoint = torch.load(last_checkpoint_path)
            # Repair the checkpoint
            checkpoint["state_dict"] = {
                k.replace("._orig_mod.", "."): v
                for k, v in checkpoint["state_dict"].items()
            }
            # Save the repaired checkpoint
            torch.save(checkpoint, last_checkpoint_path)
            log_if_rank_zero(logger, "Checkpoint saved successfully!")
        else:
            log_if_rank_zero(logger, "No checkpoint found to rename.")

    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
