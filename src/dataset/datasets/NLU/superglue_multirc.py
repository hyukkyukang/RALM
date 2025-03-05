import logging
from functools import cached_property
from typing import *

from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.dataset.datasets.base_dataset import BaseDataset
from src.retrieval.retriever import Retriever
from src.tokenization import ReLlamaTokenizer

logger = logging.getLogger("SuperGLUEMultiRCDataset")


def text_to_text_transform_for_multirc(example: Dict[str, Any]) -> Dict[str, Any]:
    # MultiRC has a passage with multiple questions, each with multiple answers
    # We'll flatten this structure for text-to-text format
    
    passage_text = example["passage"]["text"]
    questions = example["passage"]["questions"]
    
    # We'll create a separate example for each question-answer pair
    transformed_examples = []
    
    for question in questions:
        question_text = question["question"]
        
        for answer in question["answers"]:
            answer_text = answer["text"]
            
            # Create the context
            context = f"Passage: {passage_text}\nQuestion: {question_text}\nIs the following answer correct? {answer_text}\nAnswer:"
            
            # Get the label (0 or 1)
            if "label" in answer:
                target = "Yes" if answer["label"] == 1 else "No"
            else:
                # For test set where labels might not be available
                target = "Unknown"
            
            transformed_examples.append({
                "text": f"{context} {target}",
                "context": context,
                "target": target,
                "choices": ["Yes", "No"],
                "idx": answer.get("idx", 0),
                "question_idx": question.get("idx", 0),
                "passage_idx": example.get("idx", 0)
            })
    
    return transformed_examples

class SuperGLUEMultiRCDataset(BaseDataset):
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
    def collator(self) -> "MultiRCDataCollator":
        return MultiRCDataCollator(tokenizer=self.tokenizer)

    def run_pre_processing(self) -> None:
        """We convert the task into text-to-text format.
        The input is a passage, a question, and an answer, and the output is whether the answer is correct (Yes or No).
        """
        # Apply the transformation to all examples and flatten the results
        # This is different from other datasets because each example generates multiple transformed examples
        transformed_data = []
        for example in self.raw_data:
            transformed_examples = text_to_text_transform_for_multirc(example)
            transformed_data.extend(transformed_examples)
        
        # Create a new dataset from the transformed data
        from datasets import Dataset as HFDataset
        self.raw_data = HFDataset.from_list(transformed_data)
        
        return None

    def _tokenization_fn(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        return self.tokenizer(examples["text"], truncation=False)

    def run_post_processing(self) -> None:
        self.post_processed_data = self.tokenized_data
        return None


class MultiRCDataCollator(DataCollatorForLanguageModeling):
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
        
        # Add MultiRC-specific metadata
        if "question_idx" in examples[0]:
            batch["question_idx"] = [item["question_idx"] for item in examples]
        if "passage_idx" in examples[0]:
            batch["passage_idx"] = [item["passage_idx"] for item in examples]
        
        # TODO: Implement this for self.is_use_retrieval==True
        batch["retrieved_chunk_ids"] = None
        
        return batch 