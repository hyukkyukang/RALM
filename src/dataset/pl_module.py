import logging
import os
from functools import cached_property
from typing import *

import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.dataset.datasets import BaseDataset
from src.dataset.datasets.registry import DATASET_REGISTRY
from src.tokenization import ReLlamaTokenizer
from src.utils import log_if_rank_zero

logger = logging.getLogger("DataModule")


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: Optional[ReLlamaTokenizer] = None,
        is_test: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.is_test = is_test
        self.tokenizer: ReLlamaTokenizer = (
            ReLlamaTokenizer.from_pretrained(cfg.model.base_name)
            if tokenizer is None
            else tokenizer
        )

    @cached_property
    def train_dataset(self) -> BaseDataset | None:
        if self.is_test:
            return None
        dataset_name = self.cfg.training.train_dataset_name
        dataset_cls = DATASET_REGISTRY[dataset_name]
        return dataset_cls(
            cfg=self.cfg.dataset[dataset_name],
            global_cfg=self.cfg,
            tokenizer=self.tokenizer,
        )

    @cached_property
    def val_datasets(self) -> List[BaseDataset] | None:
        if self.is_test:
            return None
        datasets: List[BaseDataset] = []
        for task_name in self.cfg.validation.task_names:
            datasets.extend(self._get_dataset_from_task(task_name))
        return datasets

    @cached_property
    def test_datasets(self) -> List[BaseDataset] | None:
        if not self.is_test:
            return None
        datasets: List[BaseDataset] = []
        for task_name in self.cfg.testing.task_names:
            datasets.extend(self._get_dataset_from_task(task_name))
        return datasets

    def _get_dataset_from_task(self, task_name: str) -> List[BaseDataset]:
        datasets: List[BaseDataset] = []
        task_cfg = self.cfg.task[task_name]
        for dataset_name in task_cfg.dataset_names:
            dataset_cls = DATASET_REGISTRY[dataset_name]
            datasets.append(
                dataset_cls(
                    cfg=self.cfg.dataset[dataset_name],
                    global_cfg=self.cfg,
                    tokenizer=self.tokenizer,
                )
            )
        return datasets

    def _prepare_dataset(self, dataset: BaseDataset) -> None:
        """Downloads the dataset if not already present.
        This method is called only on 1 GPU in distributed training."""
        # Load raw data
        # Check if post_process_cache_path exists
        log_if_rank_zero(logger, f"Preparing {dataset.name} dataset...")
        if os.path.exists(dataset.post_process_cache_path):
            log_if_rank_zero(logger, "Loading cached post-processed dataset...")
            dataset.post_processed_data = dataset.load_from_disk(
                dataset.post_process_cache_path
            )
        else:
            # Check if tokenized_cache_path exists
            if os.path.exists(dataset.tokenized_cache_path):
                log_if_rank_zero(logger, "Loading cached tokenized dataset...")
                dataset.tokenized_data = dataset.load_from_disk(
                    dataset.tokenized_cache_path
                )
            else:
                log_if_rank_zero(logger, "Loading dataset...")
                dataset.load_dataset()

                # Pre-processing
                log_if_rank_zero(
                    logger,
                    f"Running pre-processing on the {len(dataset.raw_data)} raw data...",
                )
                dataset.run_pre_processing()

                # Tokenize the dataset and save as a cache file
                log_if_rank_zero(
                    logger,
                    f"Tokenizing {len(dataset.raw_data)} raw data...",
                )
                dataset.tokenize_data_and_save_to_disk(
                    batched=True,
                    remove_columns=dataset.cfg.remove_columns,
                )

            # Post-processing
            log_if_rank_zero(
                logger,
                f"Running post-processing on the {len(dataset.tokenized_data)} tokenized data...",
            )
            dataset.post_process_and_save_to_disk()

        log_if_rank_zero(
            logger,
            f"Total dataset size: {len(dataset)}",
        )
        return None

    def _setup_dataset(self, dataset: BaseDataset) -> None:
        """Loads and preprocesses the dataset for training.
        This method is called on every GPU in distributed training.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'. Currently unused.
        """
        if not os.path.exists(dataset.post_process_cache_path):
            raise FileNotFoundError(
                f"Tokenized dataset not found at {dataset.post_process_cache_path}. Run prepare_data first."
            )

        # Load the cached tokenized dataset instead of the raw dataset
        dataset.post_processed_data = dataset.load_from_disk(
            dataset.post_process_cache_path
        )

        log_if_rank_zero(
            logger,
            f"Loaded cached tokenized dataset with {len(dataset)} examples",
        )

        return dataset

    def prepare_data(self) -> None:
        if self.is_test:
            # Prepare the test dataset
            for test_dataset in self.test_datasets:
                self._prepare_dataset(test_dataset)
        else:
            # Prepare the train dataset
            self._prepare_dataset(self.train_dataset)

            # Prepare the val dataset
            for val_dataset in self.val_datasets:
                self._prepare_dataset(val_dataset)

        return None

    def setup(self, stage: str | None = None) -> None:
        if self.is_test:
            # Setup the test dataset
            for test_dataset in self.test_datasets:
                self._setup_dataset(test_dataset)
        else:
            # Setup the train dataset
            self._setup_dataset(self.train_dataset)

            # Setup the val datasets
            for val_dataset in self.val_datasets:
                self._setup_dataset(val_dataset)
        return None

    def train_dataloader(self) -> DataLoader | None:
        if self.is_test:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.per_device_batch_size,
            num_workers=self.cfg.training.num_workers,
            collate_fn=self.train_dataset.collator,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> List[DataLoader] | None:
        if self.is_test:
            return None
        # Create a list of DataLoaders for each validation dataset.
        val_dataloaders: List[DataLoader] = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(
                DataLoader(
                    val_dataset,
                    batch_size=self.cfg.validation.per_device_batch_size,
                    num_workers=self.cfg.validation.num_workers,
                    collate_fn=val_dataset.collator,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                )
            )
        return val_dataloaders

    def test_dataloader(self) -> List[DataLoader] | None:
        if not self.is_test:
            return None
        test_dataloaders: List[DataLoader] = []
        for test_dataset in self.test_datasets:
            test_dataloaders.append(
                DataLoader(
                    test_dataset,
                    batch_size=self.cfg.testing.per_device_batch_size,
                    num_workers=self.cfg.testing.num_workers,
                    collate_fn=test_dataset.collator,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                )
            )
        return test_dataloaders
