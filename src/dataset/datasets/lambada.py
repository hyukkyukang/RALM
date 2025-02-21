import os
from functools import cached_property
from typing import *

import hkkang_utils.file as file_utils
import torch
from datasets import Dataset
from omegaconf import DictConfig

from src.dataset.datasets.base_dataset import BaseDataset
from src.dataset.utils import split_text_into_context_and_last_word
from src.retrieval.retriever import Retriever
from src.tokenization import ReLlamaTokenizer


class LambadaDataset(BaseDataset):
    """Dataset class for the LAMBADA (Language Modeling Broadened to Account for Discourse Aspects) dataset.
    This dataset is designed to evaluate the capability of models to understand long-range dependencies.
    """

    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        tokenizer: ReLlamaTokenizer,
        tokenized_data: Optional[Dataset] = None,
        post_processed_data: Optional[Dataset] = None,
        retrieved_data: Optional[Dataset] = None,
        retriever: Optional[Retriever] = None,
    ):
        super().__init__(cfg, global_cfg, tokenizer, tokenized_data, post_processed_data, retrieved_data, retriever)

    @cached_property
    def collator(self) -> "LambadaDataCollator":
        return LambadaDataCollator(tokenizer=self.tokenizer, mlm=False)

    @property
    def tokenized_cache_path(self) -> str:
        # Get the parents' tokenized cache path
        parent_tokenized_cache_path: str = super().tokenized_cache_path
        suffix: str = "gpt2_author" if self.cfg.use_gpt2_author_data else "hf"
        return os.path.join(parent_tokenized_cache_path, suffix)

    @property
    def post_process_cache_path(self) -> str:
        return os.path.join(self.tokenized_cache_path, "lambada_post_processed")

    def load_dataset(self) -> None:
        """Loads and preprocesses the LAMBADA dataset.

        Returns:
            Dataset: Processed dataset with text split into context and last word.
        """
        if self.cfg.use_gpt2_author_data:
            # Load the raw dataset from local file
            local_file_path: str = os.path.join(
                self.global_cfg.root_dir_path,
                self.cfg.dir_name,
                self.cfg.gpt_2_author_file_name,
            )
            dataset: List[Dict[str, Any]] = file_utils.read_jsonl_file(local_file_path)
            dataset = Dataset.from_list(dataset)
        else:
            # Load the raw dataset from Hugging Face
            dataset: Dataset = super().load_dataset()
        # Save the raw data
        self.raw_data = dataset
        return dataset

    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        texts = [str(text) if text is not None else "" for text in examples["context"]]
        return self.tokenizer(texts)

    def run_pre_processing(self) -> None:
        self.raw_data = self.raw_data.map(
            lambda x: split_text_into_context_and_last_word(x["text"]),
            batched=False,
        )
        return None

    def run_post_processing(self, *args, **kwargs) -> None:
        self.post_processed_data = self.tokenized_data
        return None


class LambadaDataCollator:
    """Collator class for LAMBADA dataset that prepares data for model input.
    Note: Currently does not support batch processing."""

    def __init__(self, tokenizer: Any, mlm: bool = False) -> None:
        """Initialize the collator.

        Args:
            tokenizer: Tokenizer to be used for processing text
            mlm: Flag for masked language modeling (currently unused)
        """
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collates a list of examples, but we do not make batched tensor.
        Currently, lambada evaluation code does not support batch processing anyway.

        Args:
            examples: List of dictionaries containing the dataset examples

        Returns:
            Dict containing batched data with same keys as input examples
        """
        # Get keys of the first example
        first_example_keys = examples[0].keys()
        # Create a dictionary to store the batch data
        batch = {key: [] for key in first_example_keys}

        # Iterate over the examples and add the data to the batch
        for example in examples:
            for key in first_example_keys:
                if key == "input_ids":
                    item = torch.tensor(example[key])
                else:
                    item = example[key]
                batch[key].append(item)

        # TODO: Implement this for self.is_use_retrieval==True
        batch["retrieved_chunk_ids"] = None
        return batch
