from functools import cached_property
from typing import *

import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import DataCollatorForLanguageModeling

from src.dataset.datasets.base_dataset import BaseDataset
from src.dataset.utils import count_avg_chars_per_token_in_batch
from src.tokenization import ReLlamaTokenizer


class PintsAIDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        tokenizer: ReLlamaTokenizer,
        tokenized_data: Dataset | None = None,
    ):
        super().__init__(cfg, global_cfg, tokenizer, tokenized_data)

    @cached_property
    def collator(self) -> "PintsAIDataCollator":
        return PintsAIDataCollator(
            tokenizer=self.tokenizer, mlm=False, max_length=self.cfg.model.max_length
        )

    def _load_dataset(self) -> Dataset:
        # Filter out empty strings
        dataset = load_dataset(
            path=self.cfg.huggingface_dataset_name,
            split=self.cfg.split,
            cache_dir=self.hf_cache_dir_path,
            num_proc=8,
        )
        dataset = dataset.filter(lambda x: x["text"])
        return dataset

    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        texts = [str(text) if text is not None else "" for text in examples["text"]]
        return self.tokenizer(texts)

    def _run_post_processing(self) -> None:
        pass

class PintsAIDataCollator(DataCollatorForLanguageModeling):
    """A custom data collator for ReLLama that extends the HuggingFace DataCollatorForLanguageModeling.

    This collator adds additional functionality to:
        1. Track character counts for each sequence
    """

    def __init__(self, tokenizer: Any, mlm: Optional[bool] = False, max_length: Optional[int]=None) -> None:
        """Initialize the ReLLama data collator.

        Args:
            tokenizer: The tokenizer to use for encoding/decoding text
            mlm: Whether to use masked language modeling (default: False)
        """
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of examples. Add character counts for each sequence, after collation by the parent class.

        Args:
            examples: List of examples containing input_ids and other fields

        Returns:
            Dict containing:
                - All fields from parent collator (input_ids, labels, etc.)
                - char_counts: List of character counts for each sequence
        """
        # Truncate the input_ids and attention_mask to the max length
        if self.max_length is not None:
            for example in examples:
                example["input_ids"] = example["input_ids"][: self.max_length]
                example["attention_mask"] = example["attention_mask"][: self.max_length]

        # First apply the parent class collation
        batch = super().__call__(examples)

        # Decode the input_ids
        full_texts = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )

        # Count the characters and tokens in the batch
        avg_char_per_token, total_valid_tokens_cnt = count_avg_chars_per_token_in_batch(
            attention_masks=batch["attention_mask"],
            full_texts=full_texts,
            return_total_valid_tokens=True,
        )

        # Update batch with computed values
        batch.update(
            {
                "avg_char_per_token": torch.tensor(avg_char_per_token),
                "total_valid_tokens_cnt": torch.tensor(
                    total_valid_tokens_cnt, dtype=torch.int64
                ),
            }
        )

        return batch
