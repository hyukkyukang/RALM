import logging
import os
import sys

import torch

from src.model.utils import (
    convert_checkpoint_for_evaluation,
    convert_checkpoint_for_training,
)
from omegaconf import open_dict

logger = logging.getLogger("RepairCheckpoint")


def update_config(checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["hyper_parameters"]
    with open_dict(config):
        config.model.architecture = "small"
        config["architecture"] = {
            "small": {
                "layers": 12,
                "num_attention_heads": 12,
                "num_key_value_heads": 3,
                "hidden_size": 768,
                "intermediate_size": 3072,
            }
        }
    # Save the updated config
    logger.info(f"Saving updated config to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    if len(sys.argv) < 2:
        logger.error("Usage: python script.py <checkpoint_path>")
        sys.exit(1)
    if len(sys.argv) < 3:
        logger.error("Usage: python script.py <checkpoint_path> <mode_to_convert>")
        sys.exit(1)

    # Get the checkpoint path
    checkpoint_path = sys.argv[1]
    mode_to_convert = sys.argv[2]

    if mode_to_convert in ["eval", "evaluation"]:
        assert os.path.exists(
            checkpoint_path
        ), f"Checkpoint path {checkpoint_path} does not exist"
        convert_checkpoint_for_evaluation(checkpoint_path)
    elif mode_to_convert in ["train", "training"]:
        convert_checkpoint_for_training(checkpoint_path)
    elif mode_to_convert in ["update_config"]:
        update_config(checkpoint_path)
    else:
        logger.error("mode must be either 'eval' or 'train'")
        sys.exit(1)

    logger.info("Done!")
