import logging
import os
from functools import cached_property
from typing import *

import lightning as L
from datasets import Dataset as HuggingFaceDataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.dataset.datasets import (
    BaseDataset,
    CurationDataCollator,
    CurationDataset,
    LambadaDataCollator,
    LambadaDataset,
    PintsAIDataCollator,
    PintsAIDataset,
    WikiTextDataCollator,
    WikiTextDataset,
)
from src.tokenization import ReLlamaTokenizer
from src.utils import log_if_rank_zero

logger = logging.getLogger("DataModule")


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset: HuggingFaceDataset | None = None
        self.dataset_size: int = 0
        self.max_length: int = cfg.model.max_length
        self.tokenizer: ReLlamaTokenizer = ReLlamaTokenizer.from_pretrained(
            cfg.model.base_name
        )

    def __len__(self):
        if self.dataset is None:
            assert self.dataset_size > 0, "Dataset size not set"
            return self.dataset_size
        assert len(self.dataset) == self.dataset_size, "Dataset size mismatch"
        return len(self.dataset)

    @property
    def tokenized_dataset_path(self) -> str:
        return os.path.join(self.hf_cache_dir_path, "tokenized")

    @property
    def hf_cache_dir_path(self) -> str:
        return os.path.join(
            self.cfg._global.root_dir_path,
            self.cfg.dataset.dir_name,
            "huggingface",
            self.cfg.dataset.name,
        )

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

    @cached_property
    def data_collator(self) -> Callable:
        if self.cfg.dataset.name == "pints-ai":
            return PintsAIDataCollator(tokenizer=self.tokenizer, mlm=False)
        elif self.cfg.dataset.name == "lambada":
            return LambadaDataCollator(tokenizer=self.tokenizer, mlm=False)
        elif self.cfg.dataset.name == "wikitext":
            return WikiTextDataCollator(tokenizer=self.tokenizer, mlm=False)
        elif self.cfg.dataset.name == "curation":
            return CurationDataCollator(tokenizer=self.tokenizer, mlm=False)
        raise ValueError(
            f"Data collator for dataset {self.cfg.dataset.name} not supported"
        )

    def prepare_data(self) -> None:
        """Downloads the dataset if not already present.
        This method is called only on 1 GPU in distributed training."""
        dataset: BaseDataset = self.dataset_class(
            cfg=self.cfg, tokenizer=self.tokenizer
        )
        try:
            if not os.path.exists(dataset.tokenized_cache_path):
                # Download the dataset from the hub
                log_if_rank_zero(
                    logger,
                    f"Downloading {self.cfg.dataset.name} data ({self.cfg.dataset.split} split) into {self.hf_cache_dir_path}",
                )
                dataset.load_dataset()

                # Tokenize the dataset and save as a cache file
                log_if_rank_zero(logger, "Tokenizing dataset...")
                dataset.tokenize_data(
                    batched=True,
                    remove_columns=self.cfg.dataset.remove_columns,
                )

                # Check if the directory exists
                if not os.path.exists(dataset.tokenized_cache_path):
                    log_if_rank_zero(
                        logger,
                        f"Directory {dataset.tokenized_cache_path} does not exist. Creating it.",
                    )
                    os.makedirs(dataset.tokenized_cache_path)
                # Save the tokenized dataset
                log_if_rank_zero(
                    logger,
                    f"Saving tokenized dataset to {dataset.tokenized_cache_path}",
                )
                dataset.save_to_disk(dataset.tokenized_cache_path)

                log_if_rank_zero(
                    logger, f"Number of tokens in the dataset: {dataset.total_tokens}"
                )
                self.dataset_size = len(dataset)
            else:
                # Load the cached dataset
                train_dataset: BaseDataset = self.dataset_class.load_from_disk(
                    cfg=self.cfg,
                    tokenizer=self.tokenizer,
                    path=dataset.tokenized_cache_path,
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
        dataset: BaseDataset = self.dataset_class(
            cfg=self.cfg, tokenizer=self.tokenizer
        )
        try:
            if not os.path.exists(dataset.tokenized_cache_path):
                raise FileNotFoundError(
                    f"Tokenized dataset not found at {dataset.tokenized_cache_path}. Run prepare_data first."
                )

            # Load the cached tokenized dataset instead of the raw dataset
            self.dataset: BaseDataset = self.dataset_class.load_from_disk(
                cfg=self.cfg,
                tokenizer=self.tokenizer,
                path=dataset.tokenized_cache_path,
            )
            log_if_rank_zero(
                logger,
                f"Loaded cached tokenized dataset with {len(self.dataset)} examples",
            )

        except Exception as e:
            logger.error(f"Error setting up dataset: {str(e)}")
            raise

        # Post-processing
        self.dataset.run_post_processing()

        return None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.training.per_device_batch_size,
            num_workers=self.cfg.training.num_workers,
            collate_fn=self.data_collator,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.testing.per_device_batch_size,
            num_workers=self.cfg.testing.num_workers,
            collate_fn=self.data_collator,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
