import gc
import json
import logging
import os
import shutil
from functools import cached_property
from typing import *

import torch
import tqdm
from datasets import Dataset, concatenate_datasets
from omegaconf import DictConfig
from transformers import DataCollatorForLanguageModeling

from src.dataset.datasets.base_dataset import BaseDataset
from src.dataset.utils import count_avg_chars_per_token_in_batch
from src.tokenization import ReLlamaTokenizer
from src.utils import is_main_process, log_if_rank_zero

logger = logging.getLogger("PintsAIDataset")


class PintsAIDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        tokenizer: ReLlamaTokenizer,
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
    def collator(self) -> "PintsAIDataCollator":
        return PintsAIDataCollator(
            tokenizer=self.tokenizer,
            mlm=False,
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
        return self.tokenizer(texts, return_attention_mask=False)

    def run_pre_processing(self) -> None:
        self.raw_data = self.raw_data.filter(lambda x: x["text"])
        return None

    def run_post_processing(self) -> None:
        """
        Processes the tokenized data by:
          - Appending an EOS token to each example.
          - Concatenating tokens across examples.
          - Splitting the long stream of tokens into fixed-size chunks.

        To keep memory usage low and allow resuming after interruption,
        complete segments are flushed to disk along with a checkpoint file.

        After processing, all temporary datasets are loaded, concatenated,
        and the final dataset is saved to a temporary final directory.
        Only after a successful save are the temporary shards and checkpoint
        removed and the final directory renamed to self.post_process_cache_path.
        """
        dataset = self.tokenized_data

        # Create the cache directory if it doesn't exist.
        os.makedirs(self.post_process_cache_path, exist_ok=True)
        checkpoint_path = os.path.join(self.post_process_cache_path, "checkpoint.json")

        # Initialize or load checkpoint state.
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            processed_idx = checkpoint.get("processed_idx", 0)
            flush_counter = checkpoint.get("flush_counter", 0)
            all_new_token_ids = checkpoint.get("all_new_token_ids", [])
            tmp_token_ids = checkpoint.get("tmp_token_ids", [])
            log_if_rank_zero(
                logger,
                f"Resuming from checkpoint: processed_idx={processed_idx}, flush_counter={flush_counter}",
            )
        else:
            processed_idx = 0
            flush_counter = 0
            all_new_token_ids: List[List[int]] = []
            tmp_token_ids: List[int] = []

        # Flush threshold (number of complete segments before flushing to disk).
        dataset_flush_threshold: int = 1_000_000
        max_length: int = self.global_cfg.model.max_length
        eos_token_id: int = self.tokenizer.eos_token_id

        # Helper to update the checkpoint.
        def update_checkpoint(current_idx: int):
            checkpoint_data = {
                "processed_idx": current_idx,
                "flush_counter": flush_counter,
                "all_new_token_ids": all_new_token_ids,
                "tmp_token_ids": tmp_token_ids,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

        # Process examples; skip those already processed.
        for idx in tqdm.tqdm(
            range(processed_idx, len(dataset)),
            desc="Segmenting data",
            disable=not is_main_process(),
        ):
            example = dataset[idx]

            # Append EOS token
            token_ids = example["input_ids"] + [eos_token_id]

            # Add tokens to the temporary lists.
            tmp_token_ids.extend(token_ids)

            # Slice complete segments from the temporary token lists.
            while len(tmp_token_ids) >= max_length:
                all_new_token_ids.append(tmp_token_ids[:max_length])
                tmp_token_ids = tmp_token_ids[max_length:]

            # Flush complete segments to disk when threshold is reached.
            if len(all_new_token_ids) >= dataset_flush_threshold:
                tmp_path = os.path.join(
                    self.post_process_cache_path, f"tmp_{flush_counter}"
                )
                flush_counter += 1
                temp_dataset = Dataset.from_dict(
                    {
                        "input_ids": all_new_token_ids,
                    }
                )
                log_if_rank_zero(logger, f"Saving temporary dataset to {tmp_path}")
                temp_dataset.save_to_disk(tmp_path)
                # Reset the complete segments lists.
                all_new_token_ids = []
                del temp_dataset
                gc.collect()
                update_checkpoint(idx + 1)

        # Final flush of any remaining complete segments.
        if all_new_token_ids:
            tmp_path = os.path.join(
                self.post_process_cache_path, f"tmp_{flush_counter}"
            )
            flush_counter += 1
            temp_dataset = Dataset.from_dict(
                {
                    "input_ids": all_new_token_ids,
                }
            )
            log_if_rank_zero(logger, f"Saving final temporary dataset to {tmp_path}")
            temp_dataset.save_to_disk(tmp_path)
            all_new_token_ids = []
            del temp_dataset
            gc.collect()
            update_checkpoint(idx + 1)

        # Note: Any tokens left in tmp_token_ids (an incomplete segment) are dropped.

        # Load all flushed temporary datasets.
        temp_files = []
        for i in range(flush_counter):
            tmp_path = os.path.join(self.post_process_cache_path, f"tmp_{i}")
            assert os.path.exists(tmp_path), f"Temporary file {tmp_path} does not exist"
            temp_files.append(tmp_path)
        temp_datasets = [Dataset.load_from_disk(path) for path in temp_files]
        if len(temp_datasets) == 1:
            final_dataset = temp_datasets[0]
        else:
            final_dataset = concatenate_datasets(temp_datasets)

        # First, save the final dataset to a temporary final directory.
        log_if_rank_zero(
            logger,
            f"Saving final dataset to temporary path {self.post_process_cache_path}",
        )
        final_dataset.save_to_disk(self.post_process_cache_path)
        log_if_rank_zero(
            logger,
            f"Final dataset successfully saved to {self.post_process_cache_path}",
        )

        # Now cleanup: remove the checkpoint file and temporary shards.
        if os.path.exists(checkpoint_path):
            log_if_rank_zero(logger, f"Removing checkpoint file {checkpoint_path}")
            os.remove(checkpoint_path)
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                log_if_rank_zero(logger, f"Removing temporary file {tmp_path}")
                shutil.rmtree(tmp_path)
        log_if_rank_zero(logger, f"Cleanup Finished")

        self.post_processed_data = final_dataset


class PintsAIDataCollator(DataCollatorForLanguageModeling):
    """A custom data collator for ReLLama that extends the HuggingFace DataCollatorForLanguageModeling.

    This collator adds additional functionality to:
        1. Track character counts for each sequence
    """

    def __init__(
        self,
        tokenizer: Any,
        mlm: Optional[bool] = False,
    ) -> None:
        """Initialize the ReLLama data collator.

        Args:
            tokenizer: The tokenizer to use for encoding/decoding text
            mlm: Whether to use masked language modeling (default: False)
        """
        super().__init__(tokenizer=tokenizer, mlm=mlm)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of examples. Add character counts for each sequence, after collation by the parent class.

        Args:
            examples: List of examples containing input_ids and other fields

        Returns:
            Dict containing:
                - All fields from parent collator (input_ids, labels, etc.)
                - char_counts: List of character counts for each sequence
        """
        # Create attention mask for the final dataset
        for example in examples:
            example["attention_mask"] = [1] * len(example["input_ids"])

        # First apply the parent class collation
        batch = super().__call__(examples)

        # Prevent the data_idx from being on the GPU
        if "data_idx" in batch:
            batch["data_idx"] = batch["data_idx"].tolist()

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

        # Collate the retrieved chunk token ids
        if "retrieved_input_ids" in examples[0]:
            flatten_retrieved_input_ids = [
                item for example in examples for item in example["retrieved_input_ids"]
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
                "avg_char_per_token": torch.tensor(avg_char_per_token),
                "total_valid_tokens_cnt": torch.tensor(
                    total_valid_tokens_cnt, dtype=torch.int64
                ),
                # TODO: Implement this for self.is_use_retrieval==True
                "retrieved_input_ids": retrieved_input_ids,
                "num_retrieval_blocks": num_retrieval_blocks,
            }
        )

        return batch
