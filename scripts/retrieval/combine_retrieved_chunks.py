import logging
import os
from pathlib import Path
from typing import List

import hkkang_utils.misc as misc_utils
import hydra
import numpy as np
import tqdm
from omegaconf import DictConfig
from datasets import Dataset
from src.retrieval.utils import validate_saved_numpy_files
from src.utils import get_numpy_file_paths_in_dir

logger = logging.getLogger("Indexing")


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:

    # Retrieve all embedding file paths from the specified directory.
    dir_path = cfg.retrieval.retrieved_chunk_ids_dir_path
    all_chunk_ids_paths: List[str] = get_numpy_file_paths_in_dir(dir_path)
    # Validate saved files using file naming (order is not important for validation).
    logger.info(f"Validating saved chunk ids files in {dir_path}")
    validate_saved_numpy_files(all_chunk_ids_paths, read_file_content=False)
    logger.info("Validation of saved chunk ids files is complete.")
    # Sort the file paths by the passage index.
    all_chunk_ids_paths.sort(key=lambda path: int(Path(path).stem.split("_")[-2]))
    # Load all the numpy files into a single numpy array.
    all_chunk_ids: np.ndarray = np.concatenate(
        [np.load(path) for path in tqdm.tqdm(all_chunk_ids_paths)]
    )

    # Instead of trying to create a Dataset, save the numpy array directly
    output_path = os.path.join(dir_path, "combined_chunk_ids.npy")
    logger.info(f"Saving combined chunk ids to {output_path}")
    np.save(output_path, all_chunk_ids)
    logger.info("Saving combined chunk ids is complete.")

    # If you still need a Dataset format, create a directory for it
    dataset_path = os.path.join(dir_path, "combined_chunk_ids_dataset")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Create a dictionary with your numpy arrays (explicitly cast to int64)
    data_dict = {"chunk_ids": all_chunk_ids.astype(np.int64)}

    # Create the dataset from the dictionary (without schema parameter)
    dataset = Dataset.from_dict(data_dict)

    # Save the dataset to disk
    logger.info(f"Saving dataset to {dataset_path}")
    dataset.save_to_disk(dataset_path)
    logger.info("Saving dataset is complete.")


# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
