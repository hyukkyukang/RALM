import abc
import logging
import os
from functools import cached_property
from typing import *

import hkkang_utils.concurrent as concurrent_utils
import tqdm
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer

from src.tokenization import ReLlamaTokenizer
from src.utils import log_if_rank_zero

logger = logging.getLogger("BaseDataset")

class BaseDataset:
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        tokenizer: Union[ReLlamaTokenizer, AutoTokenizer],
        tokenized_data: Optional[Dataset] = None,
        post_processed_data: Optional[Dataset] = None,
        retrieved_data: Optional[Dataset] = None,
    ):
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.tokenizer: Union[ReLlamaTokenizer, AutoTokenizer] = tokenizer
        # Dataset objects from Hugging Face or local files
        self.raw_data: Optional[Dataset] = None
        # Dataset objects that is tokenized
        self.tokenized_data: Optional[Dataset] = tokenized_data
        # Dataset objects that is post-processed after tokenization
        self.post_processed_data: Optional[Dataset] = post_processed_data
        # Dataset objects that is retrieved after post-processing
        self.retrieved_data: Optional[Dataset] = retrieved_data
    def __len__(self):
        if self.post_processed_data is None:
            return 0
        return len(self.post_processed_data)

    def __getitem__(self, idx: int) -> Any:
        if self.post_processed_data is None:
            return None
        processed_data = self.post_processed_data[idx]
        return processed_data
        # TODO: Uncomment this when retrieval is implemented
        # if self.is_use_retrieval:
        #     assert self.retrieved_data is not None, "Retrieved data is not loaded"
        #     # Get the retrieved data
        #     retrieved_data = self.retrieved_data[idx]
        #     # Combine the post-processed data and the retrieved data
        #     data_to_return = processed_data.copy()
        #     data_to_return.update(retrieved_data)
        # else:
        #     data_to_return = processed_data
        # return data_to_return


    @cached_property
    @abc.abstractmethod
    def collator(self) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abc.abstractmethod
    def post_process_cache_path(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def run_pre_processing(self, *args, **kwargs) -> None:
        """Run pre-prcoessing on the raw data (i.e., before tokenization)
        Return None if no pre-processing is needed
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def run_post_processing(self, *args, **kwargs) -> None:
        """Run post-processing on the tokenized data (i.e., after tokenization)
        Return None if no post-processing is needed
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def is_use_retrieval(self) -> bool:
        return False
        # TODO: Uncomment this when retrieval is implemented
        # return self.global_cfg.model.name == "rellama" and self.global_cfg.model.is_use_retrieval

    @property
    def hf_cache_dir_path(self) -> str:
        return os.path.join(
            self.global_cfg.root_dir_path,
            self.cfg.dir_name,
            "huggingface",
            self.cfg.name,
        )

    @property
    def tokenized_cache_path(self) -> str:
        # Get tokenizer name from the tokenizer class
        tokenizer_name = self.tokenizer.name_or_path.replace("/", "_")
        return os.path.join(self.hf_cache_dir_path, f"{tokenizer_name}_tokenized")

    @property
    def retrieved_data_cache_path(self) -> str:
        chunk_size = self.global_cfg.model.retrieval_chunk_size
        return os.path.join(self.post_process_cache_path, f"retrieval_cache_{chunk_size}")

    @property
    def total_tokens(self) -> int:
        if not hasattr(self, "_total_tokens"):
            if self.tokenized_data is None:
                return 0
            # Get total number of items
            total_items = len(self.tokenized_data)
            # Get number of items per process
            items_per_process = total_items // 64
            # Initialize multiprocessor
            multiprocessor = concurrent_utils.MultiProcessor(num_workers=64)
            # Run count_tokens for each process
            for i in range(64):
                multiprocessor.run(
                    self._count_tokens,
                    process_idx=i,
                    data=self.tokenized_data,
                    indices=range(i * items_per_process, (i + 1) * items_per_process),
                )
            # Join all processes
            multiprocessor.join()
            # Sum the results
            self._total_tokens = sum(multiprocessor.results)
        return self._total_tokens

    def _count_tokens(self, process_idx: int, data: Dataset, indices: List[int]) -> int:
        enable_tqdm = process_idx == 0
        total_tokens = sum(
            sum(data[i]["attention_mask"])
            for i in tqdm.tqdm(indices, desc="Counting tokens", disable=not enable_tqdm)
        )
        print(f"Total tokens: {total_tokens} (Process {process_idx})")
        return total_tokens

    def load_dataset(self) -> Dataset:
        self.raw_data = hf_load_dataset(
            path=self.cfg.huggingface_dataset_name,
            name=self.cfg.subset,
            split=self.cfg.split,
            cache_dir=self.hf_cache_dir_path,
            num_proc=8,
        )

    def tokenize_data(
        self,
        batched: bool = True,
        remove_columns: List[str] = [],
    ) -> None:
        assert self.raw_data is not None, "Raw data is not loaded"
        if self.tokenized_data is None:
            # Check if remove_columns are present in the raw_data, if not remove them from the list
            remove_columns = [
                col for col in remove_columns if col in self.raw_data.column_names
            ]
            # Tokenize the data
            self.tokenized_data = self.raw_data.map(
                self._tokenization_fn,
                batched=batched,
                remove_columns=remove_columns,
                num_proc=64,
            )
        return None

    def retrieve_data(self) -> None:
        # TODO: Need to implement this here.
        num_of_chunks = self.global_cfg.model.max_length // self.global_cfg.model.input_chunk_size - 1
        dummy_chunk_ids: List[List[int]] = [list(range(0, self.global_cfg.model.retrieval_chunk_size))] * num_of_chunks
        self.retrieved_data = Dataset.from_dict(
            {
                "retrieved_chunk_ids": [
                    dummy_chunk_ids
                    for _ in tqdm.tqdm(range(len(self.post_processed_data)), desc="Creating dummy retrieval data")
                ]
            }
        )
        return None

    def tokenize_data_and_save_to_disk(
        self,
        batched: bool = True,
        remove_columns: List[str] = [],
        overwrite: bool = False,
    ) -> None:
        assert self.raw_data is not None, "Raw data is not loaded"
        if os.path.exists(self.tokenized_cache_path) and not overwrite:
            log_if_rank_zero(
                logger,
                f"Tokenized data already exists at {self.tokenized_cache_path}. Skip tokenization.",
            )
            self.tokenized_data = self.load_from_disk(self.tokenized_cache_path)
            return None

        # Tokenize the data
        self.tokenize_data(batched=batched, remove_columns=remove_columns)

        # Save to disk
        log_if_rank_zero(
            logger,
            f"Saving {len(self.tokenized_data)} tokenized examples to {self.tokenized_cache_path}",
        )
        self.save_to_disk(
            dataset=self.tokenized_data,
            path=self.tokenized_cache_path,
            overwrite=False,
        )
        return None

    def post_process_and_save_to_disk(self, overwrite: bool = False) -> None:
        assert self.tokenized_data is not None, "Tokenized data is not loaded"
        # Check if the post-processed data already exists
        path = self.post_process_cache_path
        if os.path.exists(path) and not overwrite:
            log_if_rank_zero(
                logger,
                f"Post-processed data already exists at {path}. Skip post-processing.",
            )
            # Load the post-processed data
            self.post_processed_data = self.load_from_disk(path)
            return None

        # Run post-processing
        self.run_post_processing()

        # Save to disk
        log_if_rank_zero(
            logger,
            f"Saving {len(self.post_processed_data)} post-processed examples to {path}",
        )
        self.save_to_disk(
            dataset=self.post_processed_data,
            path=path,
            overwrite=False,
        )
        return None

    def retrieve_and_save_to_disk(self, overwrite: bool = False) -> None:
        """Save the retrieved data to disk after post-processing.
        Save the data in the sub-directory of the post-processed data (i.e., self.retrieved_data_cache_path).
        The number of items should be the same as the number of items in the post-processed data.
        We will reference it by the same index of the post-processed data.
        """
        assert self.post_processed_data is not None, "Post-processed data is not loaded"
        # Check if the retrieved data already exists
        path = self.retrieved_data_cache_path
        if os.path.exists(path) and not overwrite:
            log_if_rank_zero(
                logger,
                f"Retrieved data already exists at {path}. Skip retrieval.",
            )
            self.retrieved_data: Dataset = self.load_from_disk(path)
            return None

        # Run retrieval
        self.retrieve_data()
        assert len(self.retrieved_data) == len(self.post_processed_data), \
            f"The number of retrieved data should be the same as the number of post-processed data: {len(self.retrieved_data)} != {len(self.post_processed_data)}"

        # Save to disk
        log_if_rank_zero(
            logger,
            f"Saving {len(self.retrieved_data)} retrieved examples to {path}",
        )
        self.save_to_disk(
            dataset=self.retrieved_data,
            path=path,
            overwrite=False,
        )
        return None
    
    
    def save_to_disk(
        self, dataset: Dataset, path: str, overwrite: bool = False
    ) -> None:
        # Check if the directory exists
        if os.path.exists(path) and not overwrite:
            log_if_rank_zero(
                logger, f"Directory {path} already exists. Skip saving to disk."
            )
            return None
        if os.path.exists(path):
            log_if_rank_zero(
                logger,
                f"Directory {path} already exists. Overwriting it.",
            )
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save to disk
        dataset.save_to_disk(path)
        return None

    def load_from_disk(self, path: str) -> Dataset:
        return Dataset.load_from_disk(path)
