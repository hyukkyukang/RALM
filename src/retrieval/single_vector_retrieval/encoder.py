import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
from functools import cached_property
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.retrieval.dataloader import StreamingDataLoader
from src.utils import AsyncEmbeddingSaver, is_main_process, is_torch_compile_possible


logger = logging.getLogger("Encoder")


class SentenceTransformerCorpusEncoder:
    def __init__(
        self,
        model_name: str,
        src_tokenizer_name: str,
        save_dir_path: str,
        device: torch.device = "cpu",
        enable_torch_compile: bool = True,
        chunk_size: int = 64,
        passage_size: int = 512,
    ):
        self.model_name = model_name
        self.src_tokenizer_name = src_tokenizer_name
        self.src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map="auto" if device is None else device,
            attn_implementation=None if is_torch_compile_possible() else "eager",
        )
        self.save_dir_path = save_dir_path
        self.enable_torch_compile = enable_torch_compile
        self.chunk_size = chunk_size
        self.passage_size = passage_size
        self.__post_init__()

    def __post_init__(self):
        if not os.path.exists(self.save_dir_path):
            os.makedirs(self.save_dir_path)
        # Check if torch compile is enabled.
        if self.enable_torch_compile and is_torch_compile_possible():
            logger.info("Compiling the model with torch compile...")
            self.model = torch.compile(self.model, dynamic=True)
        else:
            logger.info("Torch compile is not enabled.")

    @cached_property
    def num_chunk_per_passage(self) -> int:
        return self.passage_size // self.chunk_size

    def encode_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 100,
        data_span_start_idx: Optional[int] = None,
        data_span_end_idx: Optional[int] = None,
        num_dataloader_workers: int = 4,
    ) -> None:
        # Create a DataLoader that streams data from the dataset.
        dataloader = StreamingDataLoader(
            dataset,
            start_idx=data_span_start_idx,
            end_idx=data_span_end_idx,
            src_tokenizer=self.src_tokenizer,
            tgt_tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            batch_size=batch_size,
            num_workers=num_dataloader_workers,
        )
        logger.info(f"Saving embeddings to {self.save_dir_path}")
        # Instantiate AsyncSaver to perform disk writes asynchronously.
        async_saver = AsyncEmbeddingSaver(max_queue_size=3)

        disable_tqdm = not is_main_process()
        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                desc="Encoding dataset",
                total=len(dataloader),
                disable=disable_tqdm,
            )
        ):
            bsize: int = batch["input_ids"].shape[0] // self.num_chunk_per_passage
            # Calculate the starting index for the current batch.
            passage_idx = min(batch["passage_indices"])
            # Build the file path for this batch.
            file_path = os.path.join(
                self.save_dir_path, f"embeddings_{passage_idx}_{bsize}.npy"
            )
            # Skip batch if file already exists.
            if os.path.exists(file_path):
                continue
            # Move the batch to the model device.
            batch = {
                key: value.to(self.model.device)
                for key, value in batch.items()
                if key != "passage_indices"
            }
            # Perform the encoding.
            output = self.model(**batch)
            # CLS token representation
            embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()

            # Enqueue saving the embeddings asynchronously.
            async_saver.save(file_path, embeddings)

        # Wait for all asynchronous saves to complete.
        async_saver.close()
        return None
