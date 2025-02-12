import abc
import os
from typing import *

import tqdm
from datasets import Dataset
from omegaconf import DictConfig

from src.tokenization import ReLlamaTokenizer


class BaseDataset:
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: ReLlamaTokenizer,
        tokenized_data: Dataset | None = None,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.raw_data: Dataset | None = None
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
    def tokenized_cache_path(self) -> str:
        return os.path.join(self.hf_cache_dir_path, "tokenized")

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
    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def _load_dataset(self) -> Dataset:
        raise NotImplementedError("Subclasses must implement this method")

    def load_dataset(self) -> None:
        self.raw_data = self._load_dataset()
        return None

    def tokenize_data(
        self,
        batched: bool = True,
        remove_columns: List[str] = [],
    ) -> None:
        if self.tokenized_data is None:
            # Check if remove_columns are present in the raw_data, if not remove them from the list
            remove_columns = [
                col for col in remove_columns if col in self.raw_data.column_names
            ]
            # Tokenize the data
            self.tokenized_data = self.raw_data.map(
                self._tokenization_fn, batched=batched, remove_columns=remove_columns
            )
        return None

    def save_to_disk(self, path: str) -> None:
        self.tokenized_data.save_to_disk(path)

    @classmethod
    def load_from_disk(
        cls, cfg: DictConfig, tokenizer: ReLlamaTokenizer, path: str
    ) -> "BaseDataset":
        return cls(
            cfg=cfg, tokenizer=tokenizer, tokenized_data=Dataset.load_from_disk(path)
        )
