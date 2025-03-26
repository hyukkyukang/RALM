import logging
import os
import time

from lightning.pytorch.callbacks import Callback

from src.utils import log_if_rank_zero

logger = logging.getLogger("TimeBasedCheckpoint")

class TimeBasedCheckpoint(Callback):
    def __init__(self, save_interval_hours=6, dirpath="checkpoints"):
        super().__init__()
        self.save_interval_seconds = save_interval_hours * 3600  # Convert hours to seconds
        self.last_checkpoint_time = time.time()
        self.dirpath = dirpath

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Check if 6 hours have passed since the last checkpoint and save a new one."""
        current_time = time.time()
        if trainer.is_global_zero and current_time - self.last_checkpoint_time >= self.save_interval_seconds:
            self.last_checkpoint_time = current_time
            readable_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(current_time))
            checkpoint_path = os.path.join(self.dirpath, f"checkpoint-{readable_time}.ckpt")
            trainer.save_checkpoint(checkpoint_path)
            log_if_rank_zero(logger, f"Checkpoint saved at {checkpoint_path}")