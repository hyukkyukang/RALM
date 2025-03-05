import logging
from functools import cached_property
from typing import *

from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.dataset.datasets.base_dataset import BaseDataset
from src.retrieval.retriever import Retriever
from src.tokenization import ReLlamaTokenizer

logger = logging.getLogger("GLUECoLADataset")


def text_to_text_transform_for_cola(example: Dict[str, Any]) -> Dict[str, Any]:
    context = f"Judge whether the following sentence is grammatically correct or incorrect: {example['sentence']}\nAnswer:"
    target = "Correct" if example["label"] == 1 else "Incorrect"
    return {
        "text": f"{context} {target}",
        "context": context,
        "target": target,
        "choices": ["Correct", "Incorrect"],
    }

class GLUECoLADataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        tokenizer: Union[ReLlamaTokenizer, AutoTokenizer],
        tokenized_data: Optional[Dataset] = None,
        post_processed_data: Optional[Dataset] = None,
        retrieved_data: Optional[Dataset] = None,
        retriever: Optional[Retriever] = None,
        mode: Optional[str] = None,
        task_name: Optional[str] = None,
    ):
        super().__init__(cfg=cfg, 
                         global_cfg=global_cfg, 
                         tokenizer=tokenizer, 
                         tokenized_data=tokenized_data, 
                         post_processed_data=post_processed_data, 
                         retrieved_data=retrieved_data, 
                         retriever=retriever, 
                         mode=mode, 
                         task_name=task_name)

    @property
    def post_process_cache_path(self) -> str:
        return None

    @cached_property
    def collator(self) -> "CoLADataCollator":
        return CoLADataCollator(tokenizer=self.tokenizer)

    def run_pre_processing(self) -> None:
        """We convert the task into text-to-text format.
        The input is a sentence, and the output is a label (Correct or Incorrect).
        """
        # Apply the transformation to all examples
        self.raw_data = self.raw_data.map(text_to_text_transform_for_cola)
        return None

    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        return self.tokenizer(examples["text"], truncation=False)

    def run_post_processing(self) -> None:
        self.post_processed_data = self.tokenized_data
        return None


class GLUECoLADataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: Union[ReLlamaTokenizer, AutoTokenizer], mlm: Optional[bool] = False) -> None:
        self.tokenizer = tokenizer
        super().__init__(tokenizer=tokenizer, mlm=mlm)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Prepare the input_ids and attention_mask
        tmp = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in examples]

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

