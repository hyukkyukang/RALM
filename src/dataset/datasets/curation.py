from typing import *

import torch
import tqdm
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.dataset.datasets.base_dataset import BaseDataset
from src.dataset.utils import INVALID_TOKEN_ID
from src.tokenization import ReLlamaTokenizer


class CurationDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: Union[ReLlamaTokenizer, AutoTokenizer],
        tokenized_data: Dataset | None = None,
    ):
        super().__init__(cfg, tokenizer, tokenized_data)

    @property
    def total_tokens(self) -> int:
        """We override the total_tokens property to correctly count the tokens in the tokenized_data."""
        if not hasattr(self, "_total_tokens"):
            if self.tokenized_data is None:
                return 0
            all_target_key_prefixes = ["summary"]
            all_attention_masks: List[List[int]] = [
                mask
                for key_prefix in all_target_key_prefixes
                for mask in self.tokenized_data[f"{key_prefix}_attention_mask"]
            ]
            self._total_tokens = sum(
                sum(x) for x in tqdm.tqdm(all_attention_masks, desc="Counting tokens")
            )
        return self._total_tokens

    def _load_dataset(self) -> Dataset:
        dataset = load_dataset(
            path=self.cfg.dataset.huggingface_dataset_name,
            name=self.cfg.dataset.subset,
            split=self.cfg.dataset.split,
            cache_dir=self.hf_cache_dir_path,
            num_proc=8,
        )
        return dataset

    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        # Pre-process all texts at once
        titles = [
            "Title: " + str(text) if text is not None else ""
            for text in examples["title"]
        ]
        contents = [
            "\nContent: " + str(text) + "\nTL;DR: " if text is not None else ""
            for text in examples["article_content"]
        ]
        summaries = [
            str(text) if text is not None else "" for text in examples["summary"]
        ]

        # Tokenize each type separately
        title_tokens = self.tokenizer(
            titles,
            truncation=False,
            padding=False,
            add_special_tokens=True,  # Add special tokens only for titles
        )

        content_tokens = self.tokenizer(
            contents, truncation=False, padding=False, add_special_tokens=False
        )

        summary_tokens = self.tokenizer(
            summaries, truncation=False, padding=False, add_special_tokens=False
        )

        # Return the tokenized results
        return {
            "title_input_ids": title_tokens["input_ids"],
            "title_attention_mask": title_tokens["attention_mask"],
            "content_input_ids": content_tokens["input_ids"],
            "content_attention_mask": content_tokens["attention_mask"],
            "summary_input_ids": summary_tokens["input_ids"],
            "summary_attention_mask": summary_tokens["attention_mask"],
        }

    def run_post_processing(self) -> None:
        pass


class CurationDataCollator(DataCollatorForLanguageModeling):
    def __init__(
        self, tokenizer: Any, mlm: bool = False, model_max_length: int = 0
    ) -> None:
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.model_max_length = model_max_length
        assert self.model_max_length > 0, "model_max_length must be greater than 0"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Combine title, content, and summary into a single input
        new_examples = []
        for example in examples:
            input_ids, attention_mask = combine_inputs(example, self.model_max_length)
            new_examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )

        # First apply the parent class collation
        batch = super().__call__(new_examples)

        # Modify the labels:
        # 1. We only want to evaluate loss on the summary inputs
        for i, example in enumerate(examples):
            # Calculate the length of title + content (non-summary part)
            non_summary_length = len(example["title_input_ids"]) + len(
                example["content_input_ids"]
            )
            # Set labels to -100 for non-summary tokens (this ignores them in loss calculation)
            batch["labels"][i, :non_summary_length] = INVALID_TOKEN_ID

        # For each sequence in the batch, only decode tokens where labels != INVALID_TOKEN_ID
        # and skip the first token due to internal label shift
        total_chars_cnt = 0
        for seq_idx in range(len(batch["input_ids"])):
            # Get valid token positions (where labels != INVALID_TOKEN_ID)
            valid_positions = batch["labels"][seq_idx] != INVALID_TOKEN_ID
            # Skip the first valid token due to internal label shift
            # Find the first valid position
            first_valid_position = torch.where(valid_positions)[0][0]
            # Change the label of the first valid position to invalid token id
            batch["labels"][seq_idx, first_valid_position] = INVALID_TOKEN_ID
            # Get valid token positions again
            valid_positions = batch["labels"][seq_idx] != INVALID_TOKEN_ID
            # Get valid token IDs
            valid_tokens = batch["input_ids"][seq_idx][valid_positions]
            # Decode only valid tokens
            decoded_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
            # Add to total character count
            total_chars_cnt += len(decoded_text)

        # Update batch with computed values
        batch.update(
            {
                "total_chars_cnt": total_chars_cnt,
            }
        )
        return batch


def combine_inputs(
    example: Dict[str, Any], model_max_input: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """Combines tokenized title, content, and summary into a single sequence.

    The function maintains these priorities when truncating:
    1. Title and summary are always preserved in full
    2. Content is truncated from the end if total length exceeds model_max_input

    Args:
        example: Dictionary containing tokenized 'title', 'content', and 'summary'
                with both 'input_ids' and 'attention_mask' for each
        model_max_input: Maximum allowed length of the combined sequence

    Returns:
        Tuple of (combined_input_ids, combined_attention_mask)

    Raises:
        AssertionError: If title + summary length exceeds model_max_input
    """
    # This function combines the title, content, and summary into a single input.
    # This function has side effects: it will truncate the contents if the total length is greater than model_max_input.
    # Check the length of the inputs:
    # 1. Make sure title + summary is always less than model_max_input (Title is the lower bound on the context information)
    # 2. If the total combined length is greater than model_max_input, truncate the contents
    title_and_summary_length = sum(
        len(example[key]) for key in ["title_input_ids", "summary_input_ids"]
    )
    assert (
        title_and_summary_length <= model_max_input
    ), f"Title and summary length is greater than the model max input: {title_and_summary_length} > {model_max_input}"

    combined_length = sum(
        len(example[key])
        for key in ["title_input_ids", "content_input_ids", "summary_input_ids"]
    )
    # Find the truncation point
    num_tokens_to_truncate = combined_length - model_max_input
    # Truncate the contents
    if num_tokens_to_truncate > 0:
        example["content_input_ids"] = example["content_input_ids"][
            :-num_tokens_to_truncate
        ]
        example["content_attention_mask"] = example["content_attention_mask"][
            :-num_tokens_to_truncate
        ]

    # Combine the inputs
    combined_input = (
        example["title_input_ids"]
        + example["content_input_ids"]
        + example["summary_input_ids"]
    )
    combined_attention_mask = (
        example["title_attention_mask"]
        + example["content_attention_mask"]
        + example["summary_attention_mask"]
    )
    return combined_input, combined_attention_mask
