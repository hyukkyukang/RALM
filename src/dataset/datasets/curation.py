import logging
import os
from functools import cached_property
from typing import *

import torch
import tqdm
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.dataset.datasets.base_dataset import BaseDataset
from src.dataset.utils import INVALID_TOKEN_ID, perform_sliding_window_segmentation
from src.tokenization import ReLlamaTokenizer
from src.utils import is_main_process, log_if_rank_zero

logger = logging.getLogger("CurationDataset")


class CurationDataset(BaseDataset):
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
    def collator(self) -> "CurationDataCollator":
        return CurationDataCollator(
            tokenizer=self.tokenizer,
            mlm=False,
            model_max_length=self.global_cfg.model.max_length,
        )

    @property
    def post_process_cache_path(self) -> str:
        window: int = self.global_cfg.model.max_length
        stride: int = self.global_cfg.task.next_token_prediction.stride
        return os.path.join(
            self.tokenized_cache_path, f"segment_cache_{window}_{stride}"
        )

    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        # Define prefixes
        TITLE_PREFIX = "Title: "
        CONTENTS_PREFIX = "\nContent: "
        TLDR_PREFIX = "\nTL;DR: "

        # Combine title, content, and summary into a single text
        example_num = len(examples["title"])
        non_summary_texts: List[str] = []
        summary_texts: List[str] = []
        for i in range(example_num):
            title = examples["title"][i]
            title = "" if title is None else title
            content = examples["article_content"][i]
            content = "" if content is None else content
            summary = examples["summary"][i]
            summary = "" if summary is None else summary
            non_summary_text = (
                TITLE_PREFIX + title + CONTENTS_PREFIX + content + TLDR_PREFIX
            )
            summary_text = summary
            non_summary_texts.append(non_summary_text)
            summary_texts.append(summary_text)

        # Tokenize each text separately
        non_summary_tokens = self.tokenizer(
            non_summary_texts,
            truncation=False,
            padding=False,
            add_special_tokens=True,  # Add special tokens only for titles
        )
        summary_tokens = self.tokenizer(
            summary_texts,
            truncation=False,
            padding=False,
            add_special_tokens=False,  # Add special tokens only for titles
        )

        # Return the tokenized results
        return {
            "non_summary_input_ids": non_summary_tokens["input_ids"],
            "non_summary_attention_mask": non_summary_tokens["attention_mask"],
            "summary_input_ids": summary_tokens["input_ids"],
            "summary_attention_mask": summary_tokens["attention_mask"],
        }

    def run_pre_processing(self) -> None:
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

        # Perform the sliding window segmentation with caching
        segments_list: List[Dict[str, Any]] = []
        total_data_num = len(self.tokenized_data["non_summary_input_ids"])
        for i in tqdm.tqdm(
            range(total_data_num),
            desc="Segmenting data",
            disable=not is_main_process(),
        ):
            # Combine the non-summary and summary token ids
            concatenated_token_ids = (
                self.tokenized_data[i]["non_summary_input_ids"]
                + self.tokenized_data[i]["summary_input_ids"]
            )
            # Get the valid token start index
            valid_token_start_idx = len(self.tokenized_data[i]["non_summary_input_ids"])
            # Perform the sliding window segmentation
            segments: List[Dict[str, Any]] = perform_sliding_window_segmentation(
                concatenated_token_ids,
                window_size,
                stride,
                valid_token_start_idx=valid_token_start_idx,
            )

            # Filter out the segments whose labels are all masked out
            segments = [
                segment
                for segment in segments
                if not torch.all(segment["labels"] == INVALID_TOKEN_ID)
            ]

            # Aggregate the segments to the list
            segments_list.extend(segments)

        log_if_rank_zero(
            logger,
            f"Post-processing: Segmented data into {len(segments_list)} segments",
        )

        # Transform list of dicts into dict of lists
        keys_in_segments = segments_list[0].keys()
        dict_of_lists = {
            key: [example[key] for example in segments_list] for key in keys_in_segments
        }
        dataset_of_segments = Dataset.from_dict(dict_of_lists)

        # Save the segmented data
        self.post_processed_data = dataset_of_segments
        return None


class CurationDataCollator(DataCollatorForLanguageModeling):
    def __init__(
        self, tokenizer: Any, mlm: bool = False, model_max_length: int = 0
    ) -> None:
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.model_max_length = model_max_length
        assert self.model_max_length > 0, "model_max_length must be greater than 0"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assumption:
        The inputs are segmented as a sliding window and they have same length (Following the experiments from GPT-2 paper).
        The last segment maybe padded with special tokens.
        """
        # Stack the examples with torch.stack
        keys_to_stack = ["input_ids", "attention_mask", "labels"]

        # Find max length in the batch
        max_length = max(len(example["input_ids"]) for example in examples)
        max_length = min(max_length, self.model_max_length)  # Cap at model_max_length

        # Initialize padded tensors
        pad_values = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
            "labels": INVALID_TOKEN_ID,
            "retrieved_chunk_token_ids": self.tokenizer.pad_token_id,
        }
        batch = {
            key: torch.full(
                (len(examples), max_length),
                pad_values[key],
                dtype=torch.long,
                device="cpu",
            )
            for key in keys_to_stack
        }

        # Fill in the actual values
        for i, example in enumerate(examples):
            for key in keys_to_stack:
                length = len(example[key])
                batch[key][i, :length] = torch.tensor(
                    example[key][:max_length], device="cpu"
                )

        # For each sequence in the batch, only decode tokens where labels != INVALID_TOKEN_ID
        # and skip the first token due to internal label shift
        total_chars_cnt = 0
        for seq_idx in range(len(batch["input_ids"])):
            valid_positions = batch["labels"][seq_idx] != INVALID_TOKEN_ID
            valid_positions[0] = False
            valid_tokens = batch["input_ids"][seq_idx][valid_positions]
            decoded_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
            total_chars_cnt += len(decoded_text)

        # Collate the retrieved chunk token ids
        if "retrieved_chunk_token_ids" in examples[0]:
            retrieved_chunk_token_ids = torch.tensor(
                [example["retrieved_chunk_token_ids"] for example in examples],
                dtype=torch.long,
                device="cpu",
            )
        else:
            retrieved_chunk_token_ids = None

        # Update batch with computed values
        batch.update(
            {
                "total_chars_cnt": total_chars_cnt,
                # TODO: Implement this for self.is_use_retrieval==True
                "retrieved_chunk_token_ids": retrieved_chunk_token_ids,
            }
        )

        return batch
