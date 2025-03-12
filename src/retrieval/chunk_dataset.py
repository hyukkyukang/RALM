import copy
import json
import logging
import os
from pathlib import Path
from typing import *

import numpy as np
import tqdm
from datasets import Dataset
from omegaconf import DictConfig

from src.retrieval.utils import (
    convert_global_chunk_id_to_passage_id_and_local_chunk_range,
    convert_passage_id_to_global_chunk_ids,
    validate_saved_numpy_files,
)
from src.tokenization import ReLlamaTokenizer
from src.utils import get_numpy_file_paths_in_dir

logger = logging.getLogger("RetrievedChunkDataset")


class RetrievedChunkDataset(Dataset):
    def __init__(self, dataset_name: str, cfg: DictConfig, global_cfg: DictConfig):
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.dataset_name = dataset_name
        self.__post_init__()

    def __post_init__(self):
        # Load the individual shard chunk ids and combine them into .
        if not os.path.exists(self.meta_file_path):
            self._combine_distributed_chunk_ids_to_shards()
        # Load the datasets into memory.
        logger.info("Loading datasets into memory...")
        self.data
        # Load the corpus into memory.
        logger.info("Loading corpus into memory...")
        self.corpus
        return None

    def __len__(self) -> int:
        return sum(len(data) for data in self.data)

    def __getitem__(self, idx: Union[int, slice]) -> Union[List[int], List[List[int]]]:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        passage_idx = idx
        retrieved_input_ids: List[List[List[int]]] = (
            self.get_retrieved_input_ids_by_passage_id(passage_idx)
        )
        return retrieved_input_ids

    @property
    def num_chunks_to_use_per_idx(self) -> int:
        return self.global_cfg.model.retrieval_chunk_num

    @property
    def num_chunks_per_passage(self) -> int:
        return (
            self.global_cfg.model.max_length // self.global_cfg.model.input_chunk_size
        )

    @property
    def meta_file_path(self) -> str:
        return os.path.join(
            self.global_cfg.root_dir_path,
            self.cfg.dir_path,
            self.cfg.retrieved_chunk_ids_dir,
            self.dataset_name,
            self.cfg.meta_file_name,
        )

    @property
    def chunk_ids_shard_paths(self) -> List[str]:
        """Shard is the partitioned data after combining the distributed chunk ids."""
        return self.meta_data["chunk_ids_shard_paths"]

    @property
    def meta_data(self) -> Dict[str, Any]:
        if not hasattr(self, "_meta_data"):
            meta_file = json.load(open(self.meta_file_path))
            self._meta_data = meta_file
        return self._meta_data

    @property
    def data(self) -> List[Dataset]:
        if not hasattr(self, "_data"):
            # Load all shards.
            shard_paths = self.chunk_ids_shard_paths
            shards: List[Dataset] = [
                Dataset.load_from_disk(path) for path in tqdm.tqdm(shard_paths)
            ]
            self._data = shards
        return self._data

    @property
    def corpus(self) -> Dataset:
        if not hasattr(self, "_corpus"):
            # To avoid recursive loading of the entangled datasets: RetrievedChunkDataset and BaseDataset
            global_cfg = copy.deepcopy(self.global_cfg)
            global_cfg.model.is_use_retrieval = False

            tokenizer = ReLlamaTokenizer.from_pretrained(global_cfg.model.base_name)
            from src.dataset import DATASET_REGISTRY

            corpus = DATASET_REGISTRY[self.cfg.corpus_name](
                cfg=global_cfg.dataset[self.cfg.corpus_name],
                global_cfg=global_cfg,
                tokenizer=tokenizer,
            )
            corpus.post_processed_data = corpus.load_from_disk(
                corpus.post_process_cache_path
            )
            self._corpus = corpus
        return self._corpus

    def _combine_distributed_chunk_ids_to_shards(self) -> None:
        """
        Combines distributed chunk IDs into shards of fixed size.

        This method:
        1. Loads all chunk ID files from the configured directory
        2. Validates the files to ensure they're properly formatted
        3. Concatenates all chunk IDs into a single array
        4. Splits the array into shards of size self.cfg.shard_size
        5. Saves each shard as a Dataset object
        6. Creates a metadata file with paths to all shards
        """
        dir_path: str = os.path.join(
            self.global_cfg.root_dir_path,
            self.cfg.dir_path,
            self.cfg.retrieved_chunk_ids_dir,
            self.dataset_name,
        )
        all_chunk_ids_paths: List[str] = get_numpy_file_paths_in_dir(dir_path)

        # Validate saved files using file naming (order is not important for validation)
        logger.info(f"Validating saved chunk ids files in {dir_path}")
        validate_saved_numpy_files(all_chunk_ids_paths, read_file_content=False)
        logger.info("Validation of saved chunk ids files is complete.")

        # Sort the file paths by the passage index
        all_chunk_ids_paths.sort(key=lambda path: int(Path(path).stem.split("_")[-2]))

        # Load all the numpy files into a single numpy array
        all_chunk_ids: np.ndarray = np.concatenate(
            [
                np.load(path)
                for path in tqdm.tqdm(
                    all_chunk_ids_paths, desc="Loading chunk ID files"
                )
            ]
        )

        # Cast the chunk ids to int32 for memory efficiency
        assert (
            np.max(all_chunk_ids) < np.iinfo(np.int32).max
        ), "Value of chunk ids are too large to cast to int32."
        all_chunk_ids = all_chunk_ids.astype(np.int32)

        # Calculate how many shards we'll need
        shard_size: int = self.cfg.shard_size
        total_num_shards: int = len(all_chunk_ids) // shard_size + (
            1 if len(all_chunk_ids) % shard_size > 0 else 0
        )

        # Initialize containers for metadata
        shard_paths: List[str] = []
        num_items_per_shard: List[int] = []

        logger.info(
            f"Splitting chunk ids into shards of size {shard_size}. Total number of shards: {total_num_shards}"
        )

        # Create and save each shard
        for shard_idx in tqdm.tqdm(range(total_num_shards), desc="Creating shards"):
            # Extract chunk IDs for this shard
            shard_chunk_ids: np.ndarray = all_chunk_ids[
                shard_idx * shard_size : (shard_idx + 1) * shard_size
            ]

            # Create a dictionary with the chunk IDs
            data_dict: Dict[str, np.ndarray] = {"chunk_ids": shard_chunk_ids}

            # Create the dataset from the dictionary
            dataset: Dataset = Dataset.from_dict(data_dict)

            # Save the shard to a file
            shard_path: str = os.path.join(dir_path, f"chunk_ids_shard_{shard_idx}")
            dataset.save_to_disk(shard_path)

            # Track metadata
            shard_paths.append(shard_path)
            num_items_per_shard.append(len(dataset))

        # Save the shard paths and item counts to the meta file
        logger.info(f"Saving meta data to {self.meta_file_path}")
        meta_data: Dict[str, Union[List, Dict]] = {
            "chunk_ids_shard_paths": shard_paths,
            "num_items_per_shard": {
                str(idx): num_items for idx, num_items in enumerate(num_items_per_shard)
            },
            "total_num_items": sum(num_items_per_shard),
            "num_retrieved_chunk_per_item": shard_chunk_ids.shape[1],
        }
        with open(self.meta_file_path, "w") as f:
            json.dump(meta_data, f, indent=4)

        return None

    def get_chunk_token_ids_by_global_chunk_id(self, global_chunk_id: int) -> List[int]:
        """
        Get the chunk token ids by the global chunk id.
        """
        # Convert the chunk ids to global chunk ids.
        passage_id, local_chunk_start_idx, local_chunk_end_idx = (
            convert_global_chunk_id_to_passage_id_and_local_chunk_range(
                global_chunk_id,
                self.num_chunks_per_passage,
                self.global_cfg.model.input_chunk_size,
            )
        )
        # Get the actual token ids for the retrieved chunk ids
        chunk_token_ids: List[int] = self.corpus[passage_id]["input_ids"][
            local_chunk_start_idx:local_chunk_end_idx
        ]
        return chunk_token_ids

    def get_retrieved_input_ids_by_global_chunk_id(
        self, global_chunk_id: int
    ) -> List[int]:
        """
        Get the retrieved input ids for the given global chunk id.
        """
        shard_idx = global_chunk_id // self.cfg.shard_size
        shard_idx = min(shard_idx, len(self.data) - 1)
        shard = self.data[shard_idx]
        local_idx = global_chunk_id % self.cfg.shard_size
        # Get the retrieved chunk ids for the item at index idx.
        chunk_ids: List[int] = shard[local_idx]["chunk_ids"][
            : self.num_chunks_to_use_per_idx
        ]
        # Get the actual token ids for the retrieved chunk ids
        chunk_token_ids: List[List[int]] = []
        for idx, chunk_id in enumerate(chunk_ids):
            if chunk_id == -1:
                # Check remaining chunk ids are all -1
                assert all(
                    chunk_id == -1 for chunk_id in chunk_ids[idx:]
                ), "Chunk ids are not properly formatted."
                break
            chunk_token_ids.append(
                self.get_chunk_token_ids_by_global_chunk_id(chunk_id)
            )
        return chunk_token_ids

    def get_retrieved_input_ids_by_passage_id(self, passage_id: int) -> List[List[int]]:
        """
        Get the retrieved input ids for the given passage id.
        """
        global_chunk_ids: List[int] = convert_passage_id_to_global_chunk_ids(
            passage_id, self.num_chunks_per_passage
        )
        retrieved_input_ids: List[List[List[int]]] = []
        for global_chunk_id in global_chunk_ids:
            tmp = self.get_retrieved_input_ids_by_global_chunk_id(global_chunk_id)
            if len(tmp) > 0:
                retrieved_input_ids.append(tmp)
        # Remove the last item as it cannot be used as a context
        retrieved_input_ids = retrieved_input_ids[:-1]
        return retrieved_input_ids
