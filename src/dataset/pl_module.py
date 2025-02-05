from functools import cached_property
from typing import *

import lightning as L
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import logging

logger = logging.getLogger("RETRODataModule")

class RETRODataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
    
    @property
    def max_length(self):
        return self.cfg.model.max_length
    
    @property
    def batch_size(self):
        return self.cfg.trainer.batch_size

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.cfg.model.base_name)
        
    def prepare_data(self) -> None:
        """Downloads the dataset if not already present.
        This method is called only on 1 GPU in distributed training."""
        logger.info(f"Preparing data for {self.cfg.dataset.name} split {self.cfg.dataset.split}")
        load_dataset(self.cfg.dataset.name, split=self.cfg.dataset.split)

    def setup(self, stage: str | None = None) -> None:
        """Loads and preprocesses the dataset for training.
        This method is called on every GPU in distributed training.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'. Currently unused.
        """
        dataset = load_dataset(self.cfg.dataset.name, split=self.cfg.dataset.split)
        logger.info(f"Dataset loaded with {len(dataset)} examples. Tokenizing...")
        
        # Get columns to remove (only those that exist)
        columns_to_remove = [col for col in ["source_id", "source"] if col in dataset.column_names]
        
        self.train_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=columns_to_remove,
            with_progress_bar=True
        )

    def tokenize_function(self, examples: Dict) -> Dict:
        # Handle potential None or empty strings
        texts = [str(text) if text is not None else "" for text in examples["text"]]
        return self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
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

    def val_dataloader(self) -> DataLoader | None:
        # Implement if needed
        return None

    def test_dataloader(self) -> DataLoader | None:
        # Implement if needed
        return None