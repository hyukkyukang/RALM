import logging
import os
from functools import cached_property
from typing import *

import torch
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer

from src.dataset.datasets.base_dataset import BaseDataset
from src.dataset.utils import perform_sliding_window_segmentation
from src.tokenization import ReLlamaTokenizer
from src.tokenization.utils import INVALID_TOKEN_ID
from src.utils import log_if_rank_zero

logger = logging.getLogger("WikiTextDataset")


class WikiTextDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        tokenizer: Union[ReLlamaTokenizer, AutoTokenizer],
        tokenized_data: Optional[Dataset] = None,
        post_processed_data: Optional[Dataset] = None,
    ):
        super().__init__(
            cfg,
            global_cfg,
            tokenizer,
            tokenized_data,
            post_processed_data,
        )

    @cached_property
    def collator(self) -> "WikiTextDataCollator":
        return WikiTextDataCollator(tokenizer=self.tokenizer)

    @property
    def post_process_cache_path(self) -> str:
        window: int = self.global_cfg.model.max_length
        stride: int = self.global_cfg.task.next_token_prediction.stride
        return os.path.join(
            self.tokenized_cache_path, f"segment_cache_{window}_{stride}"
        )

    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        return self.tokenizer(examples["text"], truncation=False)

    def run_pre_processing(self) -> None:
        # Combine all texts into a single string
        combined_text = "\n\n".join(self.raw_data["text"])
        self.raw_data = Dataset.from_dict({"text": [combined_text]})
        return None

    def run_post_processing(self) -> None:
        """
        Must call this function after tokenization, and before passing to the inference model.
        Segment the text as a sliding window.
        The last segment maybe padded with special tokens.
        Then save the segmented data into self.tokenized_data as Dataset type.
        """
        window_size = self.global_cfg.model.max_length
        stride = self.global_cfg.task.next_token_prediction.stride
        assert window_size >= stride, "Window size must be greater than stride"
        log_if_rank_zero(logger, f"Window size: {window_size}, Stride: {stride}")

        # Get the tokenized data
        assert len(self.tokenized_data) == 1, "We assume all the text is concatenated."
        token_ids = self.tokenized_data["input_ids"][0]

        # Perform the sliding window segmentation
        segmented_data: List[Dict[str, Any]] = perform_sliding_window_segmentation(
            token_ids, window_size, stride
        )

        # Transform list of dicts into dict of lists
        dict_of_lists = {
            key: [example[key] for example in segmented_data]
            for key in segmented_data[0].keys()
        }

        # Save the segmented data into self.tokenized_data
        self.post_processed_data = Dataset.from_dict(dict_of_lists)
        return None


class WikiTextDataCollator:
    def __init__(
        self, tokenizer: Union[ReLlamaTokenizer, AutoTokenizer], **kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assumption:
        The inputs are segmented as a sliding window and they have same length (Following the experiments from GPT-2 paper).
        The last segment maybe padded with special tokens.
        """
        # Check the inputs are segmented as sliding window
        assert all(
            "is_sliding_window" in example and example["is_sliding_window"]
            for example in examples
        ), "is_sliding_window must be in the examples"

        # Stack the examples with torch.stack
        keys_to_stack = ["input_ids", "attention_mask", "labels"]
        batch = {
            key: torch.tensor([example[key] for example in examples], device="cpu")
            for key in keys_to_stack
        }

        # For each sequence in the batch, only decode tokens where labels != INVALID_TOKEN_ID
        # and skip the first token due to internal label shift
        total_chars_cnt = 0
        for seq_idx in range(len(batch["input_ids"])):
            # Get valid token positions (where labels != INVALID_TOKEN_ID)
            valid_positions = batch["labels"][seq_idx] != INVALID_TOKEN_ID
            # Skip the first token due to internal label shift
            valid_positions[0] = False
            # Get valid token IDs
            valid_tokens = batch["input_ids"][seq_idx][valid_positions]
            # Decode only valid tokens
            decoded_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
            # Add to total character count
            total_chars_cnt += len(decoded_text)

        # Collate the retrieved chunk token ids
        if "retrieved_chunk_token_ids" in examples[0]:
            flatten_retrieved_input_ids = [
                item
                for example in examples
                for item in example["retrieved_chunk_token_ids"]
            ]
            retrieved_input_ids = torch.tensor(
                flatten_retrieved_input_ids,
                dtype=torch.long,
                device="cpu",
            )
        else:
            retrieved_input_ids = None

        if "num_retrieval_blocks" in examples[0]:
            num_retrieval_blocks = [
                example["num_retrieval_blocks"] for example in examples
            ]
        else:
            num_retrieval_blocks = None

        # Update batch with computed values
        batch.update(
            {
                "total_chars_cnt": total_chars_cnt,
                # TODO: Implement this for self.is_use_retrieval==True
                "retrieved_input_ids": retrieved_input_ids,
                "num_retrieval_blocks": num_retrieval_blocks,
            }
        )

        return batch
