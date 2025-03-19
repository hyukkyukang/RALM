import os
import sys

from src.model.utils import repair_checkpoint
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

    # Get the checkpoint path
    checkpoint_path = sys.argv[1]
    
    assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist"
    
    repair_checkpoint(checkpoint_path)
    logger.info("Done!")
