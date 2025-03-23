import os
import sys

from src.model.utils import convert_checkpoint_for_evaluation, convert_checkpoint_for_training
import logging

logger = logging.getLogger("RepairCheckpoint")

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
    
    if mode_to_convert == "eval":
        assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist"
        convert_checkpoint_for_evaluation(checkpoint_path)
    elif mode_to_convert == "train":
        convert_checkpoint_for_training(checkpoint_path)
    else:
        logger.error("mode must be either 'eval' or 'train'")
        sys.exit(1)
    
    logger.info("Done!")
