import logging
import os
from functools import cached_property
from typing import *

import torch
import tqdm
from datasets import Dataset
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
            tokenizer=self.tokenizer,
            mlm=False,
            max_length=self.global_cfg.model.max_length,
        )

    @property
    def post_process_cache_path(self) -> str:
        window: int = self.global_cfg.model.max_length
        stride: int = 0
        return os.path.join(
            self.tokenized_cache_path, f"segment_cache_{window}_{stride}"
        )

    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        texts = [str(text) if text is not None else "" for text in examples["text"]]
        return self.tokenizer(texts)

    def run_pre_processing(self) -> None:
        self.raw_data = self.raw_data.filter(lambda x: x["text"])
        return None

    def run_post_processing(self) -> None:
        # Load the tokenized dataset and concatenate the texts.
        # For each example, append an EOS token (with its attention mask set to 1),
        # then accumulate tokens in temporary lists.
        # Whenever the temporary list reaches at least max_length tokens,
        # split off a chunk of exactly max_length tokens and add it to the final dataset.

        # Perform the sliding window segmentation
        dataset = self.tokenized_data
        all_new_token_ids: List[List[int]] = []
        all_new_attention_masks: List[List[int]] = []
        tmp_token_ids: List[int] = []
        tmp_attention_masks: List[int] = []

        max_length: int = self.global_cfg.model.max_length
        eos_token_id: int = self.tokenizer.eos_token_id

        should_disable_tqdm = (
            torch.distributed.is_initialized() and torch.distributed.get_rank() != 0
        )
        for example in tqdm.tqdm(
            dataset, desc="Segmenting data", disable=should_disable_tqdm
        ):
            # Retrieve token ids and attention mask for the example
            token_ids = example["input_ids"]
            attention_mask = example["attention_mask"]

            # Append the EOS token at the end of each text
            token_ids = token_ids + [eos_token_id]
            attention_mask = attention_mask + [1]

            # Extend the temporary lists with the current example's tokens
            tmp_token_ids.extend(token_ids)
            tmp_attention_masks.extend(attention_mask)

            # While we have enough tokens for a complete chunk, slice them off
            while len(tmp_token_ids) >= max_length:
                all_new_token_ids.append(tmp_token_ids[:max_length])
                all_new_attention_masks.append(tmp_attention_masks[:max_length])
                tmp_token_ids = tmp_token_ids[max_length:]
                tmp_attention_masks = tmp_attention_masks[max_length:]

        # Optionally, handle any leftover tokens that don't make a full chunk.
        # For this implementation, we drop the final incomplete chunk.
        dataset_of_segments = Dataset.from_dict(
            {
                "input_ids": all_new_token_ids,
                "attention_mask": all_new_attention_masks,
            }
        )

        # Save the segmented data
        self.post_processed_data = dataset_of_segments
        return None


class PintsAIDataCollator(DataCollatorForLanguageModeling):
    """A custom data collator for ReLLama that extends the HuggingFace DataCollatorForLanguageModeling.

    This collator adds additional functionality to:
        1. Track character counts for each sequence
    """

    def __init__(
        self,
        tokenizer: Any,
        mlm: Optional[bool] = False,
        max_length: Optional[int] = None,
    ) -> None:
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
