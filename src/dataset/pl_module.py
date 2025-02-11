import logging
import os
from typing import *

import lightning as L
from datasets import Dataset as HuggingFaceDataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.dataset.collate import ReLLamaDataCollator
from src.dataset.datasets import (
    BaseDataset,
    CurationDataset,
    LambadaDataset,
    PintsAIDataset,
    WikiTextDataset,
)
from src.tokenizer import ReLlamaTokenizer
from src.utils import log_if_rank_zero

logger = logging.getLogger("ReLLamaDataModule")


class ReLLamaDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: HuggingFaceDataset | None = None
        self.dataset_size: int = 0
        self.max_length: int = cfg.model.max_length
        self.tokenizer: ReLlamaTokenizer = ReLlamaTokenizer.from_pretrained(
            cfg.model.base_name
        )

    def __len__(self):
        if self.train_dataset is None:
            assert self.dataset_size > 0, "Dataset size not set"
            return self.dataset_size
        assert len(self.train_dataset) == self.dataset_size, "Dataset size mismatch"
        return len(self.train_dataset)

    @property
    def dataset_class(self) -> Type[BaseDataset]:
        if self.cfg.dataset.name == "pints-ai":
            return PintsAIDataset
        elif self.cfg.dataset.name == "lambada":
            return LambadaDataset
        elif self.cfg.dataset.name == "wikitext":
            return WikiTextDataset
        elif self.cfg.dataset.name == "curation":
            return CurationDataset
        raise ValueError(f"Dataset {self.cfg.dataset.name} not supported")

    @property
    def hf_cache_dir_path(self) -> str:
        return os.path.join(
            self.cfg._global.root_dir_path,
            self.cfg.dataset.dir_name,
            "huggingface",
            self.cfg.dataset.name,
        )

    @property
    def tokenized_dataset_path(self) -> str:
        return os.path.join(self.hf_cache_dir_path, "tokenized")

    def prepare_data(self) -> None:
        """Downloads the dataset if not already present.
        This method is called only on 1 GPU in distributed training."""
        try:
            if not os.path.exists(self.tokenized_dataset_path):
                # Download the dataset from the hub
                log_if_rank_zero(
                    logger,
                    f"Downloading {self.cfg.dataset.name} data ({self.cfg.dataset.split} split) into {self.hf_cache_dir_path}",
                )
                dataset: BaseDataset = self.dataset_class(cfg=self.cfg)

                # Tokenize the dataset and save as a cache file
                log_if_rank_zero(logger, "Tokenizing dataset...")
                dataset.tokenize_data(
                    self.tokenize_function,
                    batched=True,
                    remove_columns=self.cfg.dataset.remove_columns,
                )

                # Check if the directory exists
                if not os.path.exists(self.tokenized_dataset_path):
                    log_if_rank_zero(
                        logger,
                        f"Directory {self.tokenized_dataset_path} does not exist. Creating it.",
                    )
                    os.makedirs(self.tokenized_dataset_path)
                # Save the tokenized dataset
                log_if_rank_zero(
                    logger, f"Saving tokenized dataset to {self.tokenized_dataset_path}"
                )
                dataset.save_to_disk(self.tokenized_dataset_path)

                log_if_rank_zero(
                    logger, f"Number of tokens in the dataset: {dataset.total_tokens}"
                )
                self.dataset_size = len(dataset)
            else:
                # Load the cached dataset
                train_dataset: BaseDataset = self.dataset_class.load_from_disk(
                    cfg=self.cfg, path=self.tokenized_dataset_path
                )
                self.dataset_size = len(train_dataset)
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise

    def setup(self, stage: str | None = None) -> None:
        """Loads and preprocesses the dataset for training.
        This method is called on every GPU in distributed training.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'. Currently unused.
        """
        try:
            if not os.path.exists(self.tokenized_dataset_path):
                raise FileNotFoundError(
                    f"Tokenized dataset not found at {self.tokenized_dataset_path}. Run prepare_data first."
                )

            # Load the cached tokenized dataset instead of the raw dataset
            self.train_dataset: BaseDataset = self.dataset_class.load_from_disk(
                cfg=self.cfg, path=self.tokenized_dataset_path
            )
            log_if_rank_zero(
                logger,
                f"Loaded cached tokenized dataset with {len(self.train_dataset)} examples",
            )

        except Exception as e:
            logger.error(f"Error setting up dataset: {str(e)}")
            raise

        return None

    def tokenize_function(self, examples: Dict) -> Dict:
        # Handle potential None or empty strings
        texts = [str(text) if text is not None else "" for text in examples["text"]]
        if self.cfg.dataset.truncate_ok:
            return self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            return self.tokenizer(texts)

    def train_dataloader(self) -> DataLoader:
        data_collator = ReLLamaDataCollator(tokenizer=self.tokenizer, mlm=False)
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.per_device_batch_size,
            num_workers=self.cfg.training.num_workers,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        data_collator = ReLLamaDataCollator(tokenizer=self.tokenizer, mlm=False)
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.testing.per_device_batch_size,
            num_workers=self.cfg.testing.num_workers,
            collate_fn=data_collator,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
