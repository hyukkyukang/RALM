import abc
import os
from typing import *

import tqdm
from datasets import Dataset
from omegaconf import DictConfig


class BaseDataset:
    def __init__(self, cfg: DictConfig, tokenized_data: Dataset | None = None):
        self.cfg = cfg
        self.raw_data: Dataset = self._load_dataset()
        self.tokenized_data: Dataset | None = tokenized_data

    def __len__(self):
        if self.tokenized_data is None:
            return 0
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> Any:
        if self.tokenized_data is None:
            return None
        return self.tokenized_data[idx]

    @property
    def hf_cache_dir_path(self) -> str:
        return os.path.join(
            self.cfg._global.root_dir_path,
            self.cfg.dataset.dir_name,
            "huggingface",
            self.cfg.dataset.name,
        )

    @property
    def total_tokens(self) -> int:
        if not hasattr(self, "_total_tokens"):
            if self.tokenized_data is None:
                return 0
            self._total_tokens = sum(
                sum(x)
                for x in tqdm.tqdm(
                    self.tokenized_data["attention_mask"], desc="Counting tokens"
                )
            )
        return self._total_tokens

    @abc.abstractmethod
    def _load_dataset(self) -> Dataset:
        raise NotImplementedError("Subclasses must implement this method")

    def tokenize_data(
        self,
        tokenization_fn: Callable,
        batched: bool = True,
        remove_columns: List[str] = [],
    ) -> None:
        if self.tokenized_data is None:
            self.tokenized_data = self.raw_data.map(
                tokenization_fn, batched=batched, remove_columns=remove_columns
            )
        return None

    def save_to_disk(self, path: str) -> None:
        self.raw_data.save_to_disk(path)

    @classmethod
    def load_from_disk(cls, cfg: DictConfig, path: str) -> "BaseDataset":
        return cls(cfg=cfg, tokenized_data=Dataset.load_from_disk(path))
