import concurrent.futures
import logging
import math
import os
import random
import warnings
from pathlib import Path
from typing import List

import faiss
import numpy as np
import tqdm
from omegaconf import DictConfig

from src.retrieval.utils import load_embedding_file, validate_saved_numpy_files

# Ignore future warnings from dependencies
warnings.simplefilter(action="ignore", category=FutureWarning)

# Configure logger for the indexer
logger = logging.getLogger("Indexer")


class SentenceTransformerCorpusIndexer:
    """
    Class to build and save a Faiss IVF-PQ index from a collection of embedding files.
    """

    def __init__(self, cfg: DictConfig, global_cfg: DictConfig) -> None:
        """
        Initialize the Indexer with configuration settings.

        Parameters:
            cfg (DictConfig): Configuration for the indexing process (e.g., dimensions, cluster numbers).
            global_cfg (DictConfig): Global configuration including paths and seeds.
        """
        self.cfg: DictConfig = cfg
        self.global_cfg: DictConfig = global_cfg
        self.__post_init__()

    def __post_init__(self) -> None:
        """
        Set up any additional initialization requirements, such as seeding the random number generator.
        """
        random.seed(self.global_cfg.seed)

    def __call__(self) -> None:
        """
        Execute the indexing process using GPU for training:
          1. Load an exact subset of embeddings for training (random sample).
          2. Create the Faiss IVF-PQ index and transfer it to GPU.
          3. Train and add all embeddings to the index on GPU in passage index order.
          4. Transfer the index back to CPU (optional) and save it.
        """
        os.makedirs(os.path.dirname(self.cfg.index_path), exist_ok=True)

        # Step 1: Create the quantizer (L2 distance)
        quantizer: faiss.IndexFlatL2 = faiss.IndexFlatL2(self.cfg.dim)

        # Retrieve all embedding file paths from the specified directory.
        all_embedding_paths: List[str] = self.get_embedding_paths(
            self.global_cfg.retrieval.embedding_dir_path
        )
        # Validate saved files using file naming (order is not important for validation).
        logger.info(
            f"Validating saved embedding files in {self.global_cfg.retrieval.embedding_dir_path}"
        )
        validate_saved_numpy_files(all_embedding_paths, read_file_content=False)

        # Calculate the total number of embeddings from file names.
        total_num_of_embeddings: int = self.get_number_of_embeddings(
            all_embedding_paths
        )
        num_of_clusters: int = int(math.sqrt(total_num_of_embeddings))
        logger.info(f"Total number of embeddings: {total_num_of_embeddings}")
        logger.info(f"Number of clusters: {num_of_clusters}")

        # Determine the exact number of embeddings to sample for training.
        sample_size: int = min(self.cfg.train_size, total_num_of_embeddings)
        logger.info(
            f"Sampling {sample_size} vectors from {total_num_of_embeddings} total embeddings"
        )

        # --- Training: Use a random sample of file paths ---
        training_paths: List[str] = all_embedding_paths.copy()
        random.shuffle(training_paths)
        sample_embeddings: np.ndarray = self.load_exact_sample_embeddings(
            training_paths, sample_size
        )
        logger.info(
            f"Loaded {sample_embeddings.shape[0]} training embeddings into one array..."
        )

        # Step 2: Create IVF-PQ index using the quantizer.
        index: faiss.IndexIVFPQ = faiss.IndexIVFPQ(
            quantizer,
            self.cfg.dim,
            num_of_clusters,
            self.cfg.num_subquantizers,
            self.cfg.bits_per_subvector,
        )

        # Transfer the CPU index to GPU.
        gpu_res = faiss.StandardGpuResources()
        device_id = 0  # Adjust if using a different GPU.

        options = faiss.GpuClonerOptions()
        options.useFloat16LookupTables = True

        gpu_index = faiss.index_cpu_to_gpu(gpu_res, device_id, index, options)

        # Step 3: Train the index on the sampled embeddings using GPU.
        logger.info(f"Training GPU index on {sample_embeddings.shape[0]} vectors")
        gpu_index.train(sample_embeddings)

        # --- Adding: Use files sorted by passage index ---
        ordered_paths: List[str] = sorted(
            all_embedding_paths, key=lambda path: int(Path(path).stem.split("_")[-2])
        )
        logger.info("Adding embeddings to index in passage order.")

        logger.info(f"Adding embeddings to index in batches of {self.cfg.batch_size}")
        batch_size: int = self.cfg.batch_size
        for i in tqdm.tqdm(
            range(0, len(ordered_paths), batch_size), desc="Adding vectors to index"
        ):
            batch_embedding_paths: List[str] = ordered_paths[i : i + batch_size]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                batch_embeddings_list: List[np.ndarray] = list(
                    executor.map(load_embedding_file, batch_embedding_paths)
                )
            batch_embeddings: np.ndarray = np.vstack(batch_embeddings_list)
            gpu_index.add(batch_embeddings)

        # (Optional) Transfer the index back to CPU for saving.
        logger.info("Transferring index back to CPU for saving...")
        index_cpu: faiss.IndexIVFPQ = faiss.index_gpu_to_cpu(gpu_index)
        logger.info(f"Saving index to {self.cfg.index_path}")
        faiss.write_index(index_cpu, self.cfg.index_path)
        logger.info("Indexing complete.")

        # Check that the total number of embeddings in the index is correct.
        index_cpu: faiss.IndexIVFPQ = faiss.read_index(self.cfg.index_path)
        if index_cpu.ntotal != total_num_of_embeddings:
            raise ValueError(
                f"Total number of embeddings in the index ({index_cpu.ntotal}) "
                f"does not match the expected number ({total_num_of_embeddings})."
            )
        return None

    def get_embedding_paths(self, dir_path: str) -> List[str]:
        """
        Retrieve a list of all .npy embedding file paths under the given directory (recursively).

        Parameters:
            dir_path (str): The root directory to search for .npy files.

        Returns:
            List[str]: A list of file paths as strings.
        """
        return [str(p) for p in Path(dir_path).glob("**/*.npy")]

    def get_number_of_embeddings(self, paths: List[str]) -> int:
        """
        Compute the total number of embeddings by summing the counts extracted from each file name.

        Parameters:
            paths (List[str]): List of embedding file paths.

        Returns:
            int: Total number of embeddings.
        """
        return sum(self.get_number_of_embedding_from_file_name(path) for path in paths)

    def get_number_of_embedding_from_file_name(self, file_name: str) -> int:
        """
        Extract the number of embeddings stored in a file based on the naming convention.
        The method assumes that the file name ends with an underscore followed by the count,
        for example: 'embeddings_100.npy' indicates that the file contains 100 embeddings.

        Parameters:
            file_name (str): The file name to extract the count from.

        Returns:
            int: The number of embeddings indicated in the file name.
        """
        num_of_passages = int(file_name.split("_")[-1].split(".")[0])
        chunk_per_passage: int = (
            self.global_cfg.model.max_length // self.global_cfg.model.input_chunk_size
        )
        return num_of_passages * chunk_per_passage

    def load_exact_sample_embeddings(
        self, file_paths: List[str], sample_size: int
    ) -> np.ndarray:
        """
        Load exactly 'sample_size' embeddings by iterating over file paths in order.

        Parameters:
            file_paths (List[str]): List of embedding file paths in the order provided.
            sample_size (int): Exact number of embeddings to load for training.

        Returns:
            np.ndarray: A 2D numpy array containing exactly 'sample_size' embeddings.
        """
        embeddings_list: List[np.ndarray] = []
        total_loaded: int = 0

        # Iterate over file paths and load embeddings until we've reached the desired sample_size
        for path in file_paths:
            arr: np.ndarray = load_embedding_file(path)
            embeddings_list.append(arr)
            total_loaded += arr.shape[0]
            if total_loaded >= sample_size:
                break

        # Stack all loaded embeddings into one 2D array
        stacked: np.ndarray = np.vstack(embeddings_list)
        # Slice the array to return exactly sample_size embeddings if more were loaded.
        return stacked[:sample_size]
