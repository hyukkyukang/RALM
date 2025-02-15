import abc
import os
from functools import cached_property
from typing import *

import torch
import tqdm
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer

from src.tokenization import ReLlamaTokenizer


class BaseDataset:
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        tokenizer: Union[ReLlamaTokenizer, AutoTokenizer],
        tokenized_data: Dataset | None = None,
    ):
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.tokenizer: Union[ReLlamaTokenizer, AutoTokenizer] = tokenizer
        self.raw_data: Dataset | None = None
        self.tokenized_data: Dataset | None = tokenized_data
        self.is_post_processed: bool = False

    def __len__(self):
        if self.tokenized_data is None:
            return 0
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> Any:
        if self.tokenized_data is None:
            return None
        return self.tokenized_data[idx]

    @cached_property
    @abc.abstractmethod
    def collator(self) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def name(self) -> str:
        return self.cfg.name

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
    def total_tokens(self) -> int:
        if not hasattr(self, "_total_tokens"):
            if self.tokenized_data is None:
                return 0
            # Check if distributed is initialized
            is_distributed = torch.distributed.is_initialized()
            should_disable_tqdm = not (
                is_distributed and torch.distributed.get_rank() == 0
            )
            # Count the number of tokens
            self._total_tokens = sum(
                sum(x)
                for x in tqdm.tqdm(
                    self.tokenized_data["attention_mask"],
                    desc="Counting tokens",
                    disable=should_disable_tqdm,
                )
            )
        return self._total_tokens

    @abc.abstractmethod
    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def _load_dataset(self) -> Dataset:
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def _run_post_processing(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def run_post_processing(self) -> None:
        if not self.is_post_processed:
            self._run_post_processing()
            self.is_post_processed = True
        return None

    def load_dataset(self) -> None:
        self.raw_data = self._load_dataset()
        return None

    def tokenize_data(
        self,
        batched: bool = True,
        remove_columns: List[str] = [],
    ) -> None:
        # TODO: Need is really slow (got slower than before). I don't think multiple processes are being used.
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

    def save_to_disk(self, path: str) -> None:
        self.tokenized_data.save_to_disk(path)

    def load_from_disk(self, path: str) -> None:
        self.tokenized_data = Dataset.load_from_disk(path)
        return None
