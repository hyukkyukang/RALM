import logging
from typing import Any, Dict, Iterator, List

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger("StreamingDataset")

# Define this outside of any method
class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: Dataset, start_idx: int, end_idx: int, src_tokenizer: PreTrainedTokenizer, tgt_tokenizer: PreTrainedTokenizer, chunk_size: int = 64) -> None:
        self.dataset = dataset
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Process each item in the dataset range
        for idx in range(self.start_idx, self.end_idx):
            # Use the correct indexing into the dataset
            yield self._process_item(self.dataset[idx])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Ensure consistent indexing with __iter__
        actual_idx = idx + self.start_idx
        if actual_idx >= self.end_idx:
            raise IndexError(f"Index {idx} out of range for dataset with size {self.end_idx - self.start_idx}")
        return self._process_item(self.dataset[actual_idx])

    def __len__(self) -> int:
        return self.end_idx - self.start_idx
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        input_ids = item["input_ids"]
        
        # Convert to tensor if it's not already
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
            
        # Split the input_ids into chunks of chunk_size
        chunks = [input_ids[i:i+self.chunk_size] for i in range(0, len(input_ids), self.chunk_size)]
        
        all_input_ids = []
        all_attention_masks = []
        
        for chunk in chunks:
            # Decode the input_ids to text using the source tokenizer
            # Consider whether you really want to skip special tokens
            text = self.src_tokenizer.decode(chunk, skip_special_tokens=True)

            # Tokenize the text using the target tokenizer
            tokenized = self.tgt_tokenizer(
                text,
                return_tensors="pt", 
                padding="max_length",  # Use max_length to ensure consistent size
                truncation=True,
                max_length=512,
                return_attention_mask=True
            )
            
            # Remove batch dimension (squeeze) before adding to list
            all_input_ids.append(tokenized["input_ids"].squeeze(0))
            all_attention_masks.append(tokenized["attention_mask"].squeeze(0))

        # Stack
        stacked_input_ids = torch.stack(all_input_ids)
        stacked_attention_masks = torch.stack(all_attention_masks)

        # Check dimensions
        if stacked_input_ids.shape[1] != 512:
            raise ValueError(f"Expected second dimension to be 512, got {stacked_input_ids.shape[1]}")

        return {
            "input_ids": stacked_input_ids,
            "attention_mask": stacked_attention_masks
        }

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
    result["input_ids"] = torch.cat(result["input_ids"], dim=0)
    result["attention_mask"] = torch.cat(result["attention_mask"], dim=0)
    
    return result