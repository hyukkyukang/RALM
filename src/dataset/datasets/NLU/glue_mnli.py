import logging
from functools import cached_property
from typing import *

from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.dataset.datasets.base_dataset import BaseDataset
from src.retrieval.retriever import Retriever
from src.tokenization import ReLlamaTokenizer

logger = logging.getLogger("GLUEMNLIDataset")


def text_to_text_transform_for_mnli(example: Dict[str, Any]) -> Dict[str, Any]:
    context = f"Classify the relationship between the given premise and hypothesis as either Entailment, Neutral, or Contradiction:\nPremise: {example['premise']}\nHypothesis: {example['hypothesis']}\nAnswer:"
    # Set the target label
    if example["label"] == 0:
        target = "Entailment"
    elif example["label"] == 1:
        target = "Neutral"
    else:
        target = "Contradiction"
    return {
        "text": f"{context} {target}",
        "context": context,
        "target": target,
        "choices": ["Entailment", "Neutral", "Contradiction"],
    }

class GLUEMNLIDataset(BaseDataset):
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
    def collator(self) -> "MNLIDataCollator":
        return MNLIDataCollator(tokenizer=self.tokenizer)

    def run_pre_processing(self) -> None:
        """We convert the task into text-to-text format.
        The input is a sentence, and the output is a label (Positive or Negative).
        """
        # Apply the transformation to all examples
        self.raw_data = self.raw_data.map(text_to_text_transform_for_mnli)
        return None

    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        # Encode and decode and encode again to make tokenization consistent.
        tokenized_texts: List[List[int]] = self.tokenizer(examples["text"], truncation=False)["input_ids"]
        texts: List[str] = self.tokenizer.batch_decode(tokenized_texts, skip_special_tokens=True)
        return self.tokenizer(texts, truncation=False)

    def run_post_processing(self) -> None:
        self.post_processed_data = self.tokenized_data
        return None


class GLUEMNLIDataCollator(DataCollatorForLanguageModeling):
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

