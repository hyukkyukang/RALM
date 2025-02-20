import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import glob
import logging
import os
from datetime import timedelta
from typing import *

import git
import hkkang_utils.misc as misc_utils
import hkkang_utils.slack as slack_utils
import hydra
import lightning as L
import psutil
import torch
import tqdm
from lightning.pytorch.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         ModelSummary)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from src.dataset import DataModule
from src.model import LightningModule
from src.model.utils import repair_checkpoint
from src.training.checkpoint import TimeBasedCheckpoint
from src.utils import add_config, log_if_rank_zero

logger = logging.getLogger("PL_Trainer")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Changing the precision of the matmul operation to high causes error when torch compile the flex attention
torch._dynamo.config.cache_size_limit = 1000



def get_total_optimization_steps(
    total_dataset_size: int,
    per_device_batch_size: int,
    num_gpus: int,
    gradient_accumulation_steps: int,
    max_epochs: int,
) -> int:
    return (
        total_dataset_size
        // (per_device_batch_size * num_gpus * gradient_accumulation_steps)
        * max_epochs
    )


def get_total_training_steps(
    total_dataset_size: int,
    per_device_batch_size: int,
    num_gpus: int,
    max_epochs: int,
) -> int:
    return total_dataset_size // (per_device_batch_size * num_gpus) * max_epochs


def run_pretraining(cfg: DictConfig) -> None:
    tag_prefix = "debug_" if cfg.is_debug else ""
    default_root_dir = os.path.join(
        cfg.root_dir_path,
        cfg.log_dir,
        f"{tag_prefix}{cfg.tag}",
    )
    # Set the precision to high for the models that support it
    # When using Flex attention inside the rellama model, 
    # the setting of the precision to high will cause an error
    if cfg.model.name != "rellama":
        torch.set_float32_matmul_precision("high")

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
    total_optimization_steps = get_total_optimization_steps(
        total_dataset_size=len(data_module.train_dataset),
        per_device_batch_size=cfg.training.per_device_batch_size,
        num_gpus=torch.cuda.device_count(),
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_epochs=cfg.training.max_epochs,
    )
    total_training_steps = get_total_training_steps(
        total_dataset_size=len(data_module.train_dataset),
        per_device_batch_size=cfg.training.per_device_batch_size,
        num_gpus=torch.cuda.device_count(),
        max_epochs=cfg.training.max_epochs,
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
    time_based_checkpoint_callback = TimeBasedCheckpoint(
        save_interval_hours=cfg.training.checkpoint_save_interval_hours,
        dirpath=default_root_dir,
    )

    # Trainer initialization with training args
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
            timeout=timedelta(hours=1), static_graph=True, gradient_as_bucket_view=True
        ),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            # ModelSummary(max_depth=-1), # Turn this on if you want to see the model architecture (i.e., the parameter names),
            checkpoint_callback,
            time_based_checkpoint_callback,
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
    is_the_main_process = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    if (
        is_the_main_process
        and cfg.use_torch_compile
        and torch.cuda.get_device_capability()[0] >= 7
    ):
        # Find all files ending with .ckpt in the default_root_dir
        ckpt_file_paths: List[str] = glob.glob(os.path.join(default_root_dir, "*.ckpt"))
        log_if_rank_zero(
            logger, f"Found {len(ckpt_file_paths)} ckpt files in {default_root_dir}"
        )
        # For all ckpt files, repair the modules
        for ckpt_file_path in tqdm.tqdm(
            ckpt_file_paths, desc="Repairing checkpoints..."
        ):
            repair_checkpoint(ckpt_file_path)
        log_if_rank_zero(
            logger, f"{len(ckpt_file_paths)} checkpoints saved successfully!"
        )
    else:
        log_if_rank_zero(logger, f"No checkpoint found to rename in {default_root_dir}")

    return None

@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Pretty string for the config
    pretty_cfg: str = OmegaConf.to_yaml(cfg)

    # Capture full command used to launch the script
    full_command = " ".join(psutil.Process( os.getpid() ).cmdline())
    number_of_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    with slack_utils.notification(
        channel="language-modeling",
        success_msg="Succeeded pretraining language model",
        error_msg="Failed pretraining language model",
        comments=[
            f"Command line: `{full_command}`\n\n",
            f"Number of GPUs: {number_of_gpus}\n\n",
            f"with the following config:\n```{pretty_cfg}```\n",
        ],
    ):
        run_pretraining(cfg)

    print("Pretraining script done!")
    return None

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
