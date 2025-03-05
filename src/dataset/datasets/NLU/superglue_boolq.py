import logging
from functools import cached_property
from typing import *

from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.dataset.datasets.base_dataset import BaseDataset
from src.retrieval.retriever import Retriever
from src.tokenization import ReLlamaTokenizer

logger = logging.getLogger("SuperGLUEBoolQDataset")


def text_to_text_transform_for_boolq(example: Dict[str, Any]) -> Dict[str, Any]:
    context = (
        f"Read the following passage carefully:\n\n"
        f"Passage: {example['passage']}\n\n"
        f"Based on the passage, answer the following yes/no question.\n\n"
        f"Question: {example['question']}\n\n"
        f"Your answer should be either 'Yes' or 'No'.\n"
        f"Answer:"
    )

    target = "Yes" if example["label"] == 1 else "No"

    return {
        "text": f"{context} {target}",
        "context": context,
        "target": target,
        "choices": ["Yes", "No"],
    }

class SuperGLUEBoolQDataset(BaseDataset):
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
        return f"{self.global_cfg.cache_dir}/superglue_boolq_{self.mode}_post_processed.arrow"

    @cached_property
    def collator(self) -> "BoolQDataCollator":
        return BoolQDataCollator(tokenizer=self.tokenizer)

    def run_pre_processing(self) -> None:
        logger.info(f"Loading SuperGLUE BoolQ dataset for {self.mode}")
        dataset = Dataset.load_from_disk(self.cfg.dataset_path)
        
        # Apply text-to-text transformation
        dataset = dataset.map(text_to_text_transform_for_boolq, remove_columns=dataset.column_names)
        self.tokenized_data = dataset

    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        return self.tokenizer(examples["text"], truncation=True, padding=False)

    def run_post_processing(self) -> None:
        logger.info(f"Post-processing SuperGLUE BoolQ dataset for {self.mode}")
        self.post_processed_data = self.tokenized_data


class BoolQDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: Union[ReLlamaTokenizer, AutoTokenizer], mlm: Optional[bool] = False) -> None:
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.tokenizer = tokenizer

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Prepare the input_ids and attention_mask
        batch = self.tokenizer.pad(
            {"input_ids": [example["input_ids"] for example in examples]},
            padding=True,
            return_tensors="pt",
        )
        
        # Create labels by copying input_ids and setting non-target tokens to -100
        labels = batch["input_ids"].clone()
        
        for i, example in enumerate(examples):
            # Get the target text
            target = example["target"]
            
            # Tokenize the target
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            
            # Find the position of the target in the input_ids
            input_ids = batch["input_ids"][i].tolist()
            
            # Find the starting position of the target in the input_ids
            # We look for the first occurrence of the first token of the target
            start_idx = None
            for j in range(len(input_ids) - len(target_ids) + 1):
                if input_ids[j:j+len(target_ids)] == target_ids:
                    start_idx = j
                    break
            
            if start_idx is not None:
                # Set all tokens before the target to -100
                labels[i, :start_idx] = -100
            else:
                # If target not found, set all to -100 (should not happen)
                labels[i, :] = -100
        
        batch["labels"] = labels
        return batch 