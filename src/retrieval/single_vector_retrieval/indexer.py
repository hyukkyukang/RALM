import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
from pathlib import Path
from typing import *

import faiss
import numpy as np
import tqdm
from omegaconf import DictConfig

logger = logging.getLogger("Indexer")

class Indexer:
    def __init__(self, cfg: DictConfig, global_cfg: DictConfig):
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.__post_init__()

    def __post_init__(self) -> None:
        random.seed(self.global_cfg.seed)

    def __call__(self) -> None:
        # Step 1: Create the quantizer
        quantizer = faiss.IndexFlatL2(self.cfg.dim)  # L2 quantizer

        # Step 2: Create IVF-PQ index
        index = faiss.IndexIVFPQ(quantizer, self.cfg.dim, self.cfg.num_pq_subquantizers, self.cfg.bits_per_subvector)

        # Randomly sample embedding paths without modifying the original list
        all_embedding_paths: List[str] = self.get_embedding_paths(self.global_cfg.encoding.save_dir_path)
        sample_size: int = min(self.cfg.train_size, len(all_embedding_paths))
        logger.info(f"Sampling {sample_size} vectors from {len(all_embedding_paths)}")
        sample_embedding_paths: List[str] = random.sample(all_embedding_paths, sample_size)
        sample_embeddings: List[np.ndarray] = [np.load(path) for path in sample_embedding_paths]
        sample_embeddings = np.array(sample_embeddings)

        # Step 3: Train on a subset (sample 1M vectors)
        logger.info(f"Training index on {len(sample_embeddings)} vectors")
        index.train(sample_embeddings)

        # Step 4: Add vectors to the index in batches
        logger.info(f"Adding {len(all_embedding_paths)} vectors to index in batches of {self.cfg.batch_size}")
        batch_size = self.cfg.batch_size
        for i in tqdm.tqdm(range(0, len(all_embedding_paths), batch_size), desc="Adding vectors to index"):
            # Load the batch of embeddings
            batch_embedding_paths: List[str] = all_embedding_paths[i:i + batch_size]
            batch_embeddings: List[np.ndarray] = [np.load(path) for path in batch_embedding_paths]
            batch_embeddings = np.array(batch_embeddings)
            index.add(batch_embeddings)

        # Save index
        logger.info(f"Saving index to {self.cfg.index_path}")
        faiss.write_index(index, self.cfg.index_path)

        logger.info("Indexing complete.")
        return None

    def get_embedding_paths(self, dir_path: str) -> List[str]:
        return list(Path(dir_path).glob("**/*.npy"))
