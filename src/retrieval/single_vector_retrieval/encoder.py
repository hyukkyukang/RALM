import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.retrieval.single_vector_retrieval.dataloader import StreamingDataset, collate_fn
                                                              
from src.utils import is_main_process, is_torch_compile_possible

logger = logging.getLogger("Encoder")


class Encoder:
    def __init__(self, model_name: str, src_tokenizer_name: str, save_dir_path: str, device: torch.device="cpu", enable_torch_compile: bool=True):
        self.model_name = model_name
        self.src_tokenizer_name = src_tokenizer_name
        self.src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map=device if device is not None else "auto", attn_implementation=None if is_torch_compile_possible() else "eager" )
        self.save_dir_path = save_dir_path
        self.enable_torch_compile = enable_torch_compile
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

        # Use the now-external StreamingDataset class
        streaming_dataset = StreamingDataset(dataset, dataset_start_idx, dataset_end_idx, self.src_tokenizer, self.tokenizer)

        # Use a lambda to call the external collate_fn with the required parameters
        dataloader = DataLoader(
            streaming_dataset, 
            batch_size=batch_size,
            num_workers=num_dataloader_workers,
            collate_fn=collate_fn
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
        disable_tqdm = not is_main_process()
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding dataset", total=len(dataloader), disable=disable_tqdm)):
            # Get the index of the first embedding in the batch.
            emb_idx = dataset_start_idx + batch_idx * batch_size
            # Get the path to the file that will store the embeddings.
            file_path = os.path.join(self.save_dir_path, f"embeddings_{emb_idx}_{bsize}.npy")
            # If the file already exists, skip the batch.
            if save_in_disk and os.path.exists(file_path):
                continue
            # Pass the pre-tokenized batch directly to the model.
            # Move the batch to the correct device.
            bsize = batch["input_ids"].shape[0]
            batch = {key: value.to(self.model.device) for key, value in batch.items()}
            output = self.model(**batch)
            embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token representation

            if save_in_disk:
                np.save(file_path, embeddings)
            else:
                all_embeddings.append(embeddings)
        
        if save_in_disk:
            return None
        else:
            # Concatenate and return all embeddings.
            return np.concatenate(all_embeddings, axis=0)