import abc
import logging
from functools import cached_property
from typing import *

from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.dataset.datasets.base_dataset import BaseDataset
from src.tokenization import ReLlamaTokenizer

logger = logging.getLogger("TextToTextDataset")


class TextToTextDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        tokenizer: Union[ReLlamaTokenizer, AutoTokenizer],
        tokenized_data: Optional[Dataset] = None,
        post_processed_data: Optional[Dataset] = None,
        task_name: Optional[str] = None,
    ):
        super().__init__(
            cfg=cfg,
            global_cfg=global_cfg,
            tokenizer=tokenizer,
            tokenized_data=tokenized_data,
            post_processed_data=post_processed_data,
            task_name=task_name,
        )

    @abc.abstractmethod
    def _preprocess_fn_for_text_to_text(
        self, example: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement this method for each task to convert the task into text-to-text format."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    @property
    def post_process_cache_path(self) -> str:
        return None

    @cached_property
    def collator(self) -> "TextToTextDataCollator":
        return TextToTextDataCollator(tokenizer=self.tokenizer)

    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        return self.tokenizer(examples["text"], truncation=False)

    def run_pre_processing(self) -> None:
        """We convert the task into text-to-text format.
        The input is a text with a pronoun and a noun, and the output is whether the pronoun refers to the noun (Yes or No).
        """
        # Apply the transformation to all examples
        self.raw_data = self.raw_data.map(self._preprocess_fn_for_text_to_text)
        return None

    def run_post_processing(self) -> None:
        self.post_processed_data = self.tokenized_data
        return None


class TextToTextDataCollator(DataCollatorForLanguageModeling):
    def __init__(
        self,
        tokenizer: Union[ReLlamaTokenizer, AutoTokenizer],
        mlm: Optional[bool] = False,
    ) -> None:
        self.tokenizer = tokenizer
        super().__init__(tokenizer=tokenizer, mlm=mlm)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Prepare the input_ids and attention_mask
        tmp = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
            for item in examples
        ]

        # Prepare the batch as for language modeling
        batch = super().__call__(tmp)

        # Append extra informations
        batch["context"] = [item["context"] for item in examples]
        batch["target"] = [item["target"] for item in examples]
        batch["choices"] = examples[0]["choices"]
        batch["indices"] = [item["idx"] for item in examples]

        # TODO: Implement this for self.is_use_retrieval==True
        batch["retrieved_chunk_ids"] = None

        return batch
