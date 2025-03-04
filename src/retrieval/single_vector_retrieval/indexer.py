import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
from pathlib import Path
from typing import *

import hkkang_utils.misc as misc_utils
import hydra
import numpy as np
import torch
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("Indexer")

class Indexer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @torch.no_grad()
    def encode(self, texts: List[str], save_path: str = None) -> Optional[np.ndarray]:
        with torch.inference_mode():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            output = self.model(**inputs)
        return output.last_hidden_state[:, 0, :].cpu().numpy()  # Extract CLS token representation  

    def encode_dataset(self, dataset: Dataset, batch_size: int = 100, save_path: str = None) -> Optional[np.ndarray]    :
        embeddings = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            texts = [row["text"] for row in batch]
            embeddings.append(self.encode(texts))
        embeddings = np.concatenate(embeddings)
        return self.encode(dataset["text"])


    def index(self, texts: List[str]):
        pass