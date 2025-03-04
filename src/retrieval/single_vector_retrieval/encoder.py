import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
from functools import cached_property
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.retrieval.single_vector_retrieval.dataloader import (StreamingDataset,
                                                              collate_fn)
from src.utils import is_main_process, is_torch_compile_possible

logger = logging.getLogger("Encoder")

import queue
import threading


class AsyncSaver:
    """
    Saves data to disk asynchronously using a background thread.
    """
    def __init__(self, max_queue_size: int = 3):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True  # Allow thread to exit if main process exits
        self.thread.start()

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:  # Sentinel to shutdown
                break
            file_path, embeddings = item
            try:
                np.save(file_path, embeddings)
            except Exception as e:
                logger.error(f"Error saving {file_path}: {e}")
            self.queue.task_done()

    def save(self, file_path: str, embeddings: np.ndarray):
        self.queue.put((file_path, embeddings))

    def close(self):
        # Signal the worker to shutdown and wait for it to finish.
        self.queue.put(None)
        self.thread.join()


class Encoder:
    def __init__(self, model_name: str, src_tokenizer_name: str, save_dir_path: str, device: torch.device="cpu", enable_torch_compile: bool=True, chunk_size: int=64, passage_size: int=512):
        self.model_name = model_name
        self.src_tokenizer_name = src_tokenizer_name
        self.src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto" if device is None else device, attn_implementation=None if is_torch_compile_possible() else  "eager")
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

    @torch.no_grad()
    def encode(self, texts: List[str], save_path: str = None) -> Optional[np.ndarray]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        output = self.model(**inputs)
        embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()  # Extract CLS token representation
        if save_path is None:
            return embeddings
        else:
            np.save(save_path, embeddings)
            return None

    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        dataset_start_idx: Optional[int] = None,
        dataset_end_idx: Optional[int] = None,
        num_dataloader_workers: int = 4
    ) -> DataLoader:
        # Use defaults if start index is not provided
        if dataset_start_idx is None:
            dataset_start_idx = 0

        # Use the external StreamingDataset class
        streaming_dataset = StreamingDataset(dataset, dataset_start_idx, dataset_end_idx, self.src_tokenizer, self.tokenizer, self.chunk_size)

        # Create the DataLoader with the external collate_fn.
        dataloader = DataLoader(
            streaming_dataset, 
            batch_size=batch_size,
            num_workers=num_dataloader_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        return dataloader

    @torch.no_grad()
    def encode_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 100,
        dataset_start_idx: Optional[int] = None,
        dataset_end_idx: Optional[int] = None,
        save_in_disk: bool = True,
        num_dataloader_workers: int = 4
    ) -> Optional[np.ndarray]:
        # Create a DataLoader that streams data from the dataset.
        dataloader = self.create_dataloader(dataset, batch_size, dataset_start_idx, dataset_end_idx, num_dataloader_workers)
        all_embeddings: List[np.ndarray] = []
        if save_in_disk:
            logger.info(f"Saving embeddings to {self.save_dir_path}")
            # Instantiate AsyncSaver to perform disk writes asynchronously.
            async_saver = AsyncSaver(max_queue_size=3)
        disable_tqdm = not is_main_process()
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding dataset", total=len(dataloader), disable=disable_tqdm)):
            bsize: int = batch["input_ids"].shape[0] // self.num_chunk_per_passage
            # Calculate the starting index for the current batch.
            passage_idx = batch["passage_idx"]
            # Build the file path for this batch.
            file_path = os.path.join(self.save_dir_path, f"embeddings_{passage_idx}_{bsize}.npy")
            # Skip batch if file already exists.
            if save_in_disk and os.path.exists(file_path):
                continue
            # Move the batch to the model device.
            batch = {key: value.to(self.model.device) for key, value in batch.items() if key != "passage_idx"}
            output = self.model(**batch)
            embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token representation

            if save_in_disk:
                # Enqueue saving the embeddings asynchronously.
                async_saver.save(file_path, embeddings)
            else:
                all_embeddings.append(embeddings)
        
        if save_in_disk:
            # Wait for all asynchronous saves to complete.
            async_saver.close()
            return None
        else:
            # Concatenate and return embeddings.
            return np.concatenate(all_embeddings, axis=0)