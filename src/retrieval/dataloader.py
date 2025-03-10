import logging
from typing import Any, Callable, Dict, Iterator, List

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger("StreamingDataset")


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Initialize the output dictionary
    result = {"input_ids": [], "attention_mask": []}

    # assert passage_idx are continuous
    assert all(
        item["passage_idx"] == batch[0]["passage_idx"] + i
        for i, item in enumerate(batch)
    ), f"Passage idx are not continuous"

    # Compute the number of chunks in each passage
    num_chunks_per_passage = len(batch[0]["input_ids"])
    passage_indices: List[int] = []
    for example in batch:
        passage_indices.extend([example["passage_idx"]] * num_chunks_per_passage)

    # Collect all tensors from the batch
    for item in batch:
        result["input_ids"].append(item["input_ids"])
        result["attention_mask"].append(item["attention_mask"])

    # Stack tensors into batches
    result["input_ids"] = torch.cat(result["input_ids"], dim=0)
    result["attention_mask"] = torch.cat(result["attention_mask"], dim=0)
    result["passage_indices"] = passage_indices
    return result


class StreamingCorpusDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        start_idx: int,
        end_idx: int,
        src_tokenizer: PreTrainedTokenizer,
        tgt_tokenizer: PreTrainedTokenizer,
        chunk_size: int,
        batch_size: int,
        num_workers: int = 0,
        collate_fn: Callable = collate_fn,
        pin_memory: bool = True,
        **kwargs,
    ) -> None:
        # Create a StreamingDataset
        dataset = StreamingDataset(
            dataset,
            start_idx,
            end_idx,
            src_tokenizer,
            tgt_tokenizer,
            chunk_size,
        )
        # Initialize the DataLoader
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            **kwargs,
        )


class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        start_idx: int,
        end_idx: int,
        src_tokenizer: PreTrainedTokenizer,
        tgt_tokenizer: PreTrainedTokenizer,
        chunk_size: int = 64,
    ) -> None:
        """
        Args:
            dataset (Dataset): Hugging Face Dataset.
            start_idx (int): Start index of the dataset partition.
            end_idx (int): End index of the dataset partition.
            src_tokenizer (PreTrainedTokenizer): Tokenizer for source text.
            tgt_tokenizer (PreTrainedTokenizer): Tokenizer for target text.
            chunk_size (int, optional): Chunk size (default: 64).
        """
        super().__init__()
        self.dataset = dataset
        self.start_idx = start_idx
        self.end_idx = min(
            end_idx, len(dataset)
        )  # Ensure end_idx does not exceed dataset size
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.chunk_size = chunk_size

        self.total_size = (
            self.end_idx - self.start_idx
        )  # Effective dataset size after manual partition
        self.additional_tok_length = 24

    def _partition_data(self, worker_id: int, num_workers: int):
        """
        Splits manually selected dataset range among workers.

        Args:
            worker_id (int): Worker ID.
            num_workers (int): Total number of workers.

        Returns:
            range: A range of dataset indices assigned to this worker.
        """
        per_worker = self.total_size // num_workers
        remainder = self.total_size % num_workers  # Handle uneven splits

        start = self.start_idx + worker_id * per_worker + min(worker_id, remainder)
        end = start + per_worker + (1 if worker_id < remainder else 0)

        return range(start, end)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterates over the dataset partition assigned to this worker.

        Returns:
            Iterator[Dict[str, torch.Tensor]]: Processed dataset batches.
        """
        worker_info = torch.utils.data.get_worker_info()

        # First partitioning: Manually specified start_idx to end_idx
        if worker_info is None:
            data_range = range(self.start_idx, self.end_idx)
        else:
            # Second partitioning: Distribute this range among workers
            data_range = self._partition_data(worker_info.id, worker_info.num_workers)

        logger.info(
            f"Worker {worker_info.id if worker_info else 0}: Processing indices {data_range.start} to {data_range.stop}"
        )

        for idx in data_range:
            yield self._process_item(self.dataset[idx], idx)

    def __len__(self) -> int:
        """Returns the dataset length after manual partitioning."""
        return self.total_size

    def _process_item(
        self, item: Dict[str, Any], passage_idx: int
    ) -> Dict[str, torch.Tensor]:
        input_ids = item["input_ids"]

        # Convert to tensor if it's not already
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)

        # Split the input_ids into chunks of chunk_size
        chunks = [
            input_ids[i : i + self.chunk_size]
            for i in range(0, len(input_ids), self.chunk_size)
        ]

        # Decode all chunks at once
        texts: List[str] = self.src_tokenizer.batch_decode(
            chunks, skip_special_tokens=True
        )

        # Tokenize all texts in a single batch
        tokenized: Dict[str, torch.Tensor] = self.tgt_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.chunk_size + self.additional_tok_length,
            return_attention_mask=True,
        )

        # Split the batch results into individual tensors
        all_input_ids: List[torch.Tensor] = [
            tensor for tensor in tokenized["input_ids"]
        ]
        all_attention_masks: List[torch.Tensor] = [
            tensor for tensor in tokenized["attention_mask"]
        ]

        # Stack
        stacked_input_ids = torch.stack(all_input_ids)
        stacked_attention_masks = torch.stack(all_attention_masks)

        # Check dimensions
        if stacked_input_ids.shape[1] != self.chunk_size + self.additional_tok_length:
            raise ValueError(
                f"Expected second dimension to be {self.chunk_size+self.additional_tok_length}, got {stacked_input_ids.shape[1]}"
            )

        return {
            "input_ids": stacked_input_ids,
            "attention_mask": stacked_attention_masks,
            "passage_idx": passage_idx,
        }
