from itertools import islice
from typing import Dict, Iterator, Any, List

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer


# Define this outside of any method
class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: Dataset, start_idx: int, end_idx: int, src_tokenizer: PreTrainedTokenizer, tgt_tokenizer: PreTrainedTokenizer) -> None:
        self.dataset = dataset
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Process each item in the dataset range using the same logic as __getitem__
        for idx in range(self.start_idx, self.end_idx):
            # Get the raw item
            raw_item = self.dataset[idx - self.start_idx]
            input_ids = raw_item["input_ids"]
            
            # Decode the input_ids to text using the source tokenizer
            text = self.src_tokenizer.decode(input_ids)
            
            # Tokenize the text using the target tokenizer
            tokenized = self.tgt_tokenizer(
                text,
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            )

            # Remove the batch dimension
            tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

            yield tokenized

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx = idx + self.start_idx
        input_ids = self.dataset[idx]["input_ids"]
        # Decode the input_ids to text using the source tokenizer.
        text = self.src_tokenizer.decode(input_ids)
        # Tokenize the text using the target tokenizer.
        tokenized = self.tgt_tokenizer(
            text,
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        )
        return tokenized

    def __len__(self) -> int:
        return self.end_idx - self.start_idx
    
    
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Initialize the output dictionary
    result = {
        "input_ids": [],
        "attention_mask": []
    }

    # Collect all tensors from the batch
    for item in batch:
        result["input_ids"].append(item["input_ids"])
        result["attention_mask"].append(item["attention_mask"])
    
    # Stack tensors into batches
    result["input_ids"] = torch.stack(result["input_ids"])
    result["attention_mask"] = torch.stack(result["attention_mask"])
    
    return result