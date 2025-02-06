import os
import logging
from typing import *

import lightning as L
from datasets import load_dataset, Dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from src.tokenizer import RETROTokenizer

logger = logging.getLogger("RETRODataModule")


class RETRODataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset_length = 0
        self.train_dataset: Dataset | None = None
        self.max_length: int = cfg.model.max_length
        self.batch_size: int = cfg.training.per_device_train_batch_size
        self.tokenizer: RETROTokenizer = RETROTokenizer.from_pretrained(
            cfg.model.base_name
        )

    def __len__(self):
        if self.train_dataset is None:
            return 0
        return len(self.train_dataset)

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
                logger.info(
                    f"Downloading {self.cfg.dataset.name} data ({self.cfg.dataset.split} split) into {self.hf_cache_dir_path}"
                )
                raw_dataset = load_dataset(
                    self.cfg.dataset.name,
                    split=None,
                    cache_dir=self.hf_cache_dir_path,
                    num_proc=8,
                )

                # Validate that the dataset has the required columns
                if "text" not in raw_dataset[self.cfg.dataset.split].column_names:
                    raise ValueError(
                        f"Dataset {self.cfg.dataset.name} does not contain a 'text' column"
                    )

                # Sample the dataset
                raw_dataset = raw_dataset[self.cfg.dataset.split].select(range(10000))

                # Tokenize the dataset and save as a cache file
                logger.info("Tokenizing dataset...")
                processed_dataset = raw_dataset.map(
                    self.tokenize_function,
                    batched=True,
                    remove_columns=["text", "source_id", "source"],
                )
                # Print the number of tokens in the dataset
                logger.info(
                    f"Number of tokens in the dataset: {sum(len(x) for x in processed_dataset['input_ids'])}"
                )
                # Check if the directory exists
                if not os.path.exists(self.tokenized_dataset_path):
                    logger.info(
                        f"Directory {self.tokenized_dataset_path} does not exist. Creating it."
                    )
                    os.makedirs(self.tokenized_dataset_path)
                # Save the tokenized dataset
                logger.info(
                    f"Saving tokenized dataset to {self.tokenized_dataset_path}"
                )
                processed_dataset.save_to_disk(self.tokenized_dataset_path)
            else:
                logger.info(
                    f"Loading tokenized dataset from {self.tokenized_dataset_path}"
                )
                processed_dataset = Dataset.load_from_disk(self.tokenized_dataset_path)

            self.dataset_length = processed_dataset.num_rows
            logger.info(f"Dataset length: {self.dataset_length}")

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
            self.train_dataset = Dataset.load_from_disk(self.tokenized_dataset_path)
            logger.info(
                f"Loaded cached tokenized dataset with {len(self.train_dataset)} examples"
            )

            assert (
                len(self.train_dataset) == self.dataset_length
            ), f"Dataset length mismatch: {len(self.train_dataset)} != {self.dataset_length}"

        except Exception as e:
            logger.error(f"Error setting up dataset: {str(e)}")
            raise

        return None

    def tokenize_function(self, examples: Dict) -> Dict:
        # Handle potential None or empty strings
        texts = [str(text) if text is not None else "" for text in examples["text"]]
        return self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def train_dataloader(self) -> DataLoader:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=data_collator,
            shuffle=True,
        )
