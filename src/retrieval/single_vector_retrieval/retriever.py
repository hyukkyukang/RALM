import logging
from typing import List, Tuple, Union

import faiss
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer

from src.dataset import PintsAIDataset
from src.tokenization.registry import TOKENIZER_REGISTRY
from src.utils import is_torch_compile_possible

logger = logging.getLogger("SentenceTransformerRetriever")


class SentenceTransformerRetriever:
    def __init__(self, cfg: DictConfig, global_cfg: DictConfig) -> None:
        self.cfg = cfg
        self.global_cfg = global_cfg

    @property
    def use_gpu(self) -> bool:
        return self.cfg.use_gpu and torch.cuda.is_available()

    @property
    def chunk_size(self) -> int:
        return self.cfg.encoding.chunk_size

    @property
    def num_chunks_per_passage(self) -> int:
        return self.cfg.encoding.passage_size // self.chunk_size

    @property
    def collection(self) -> List[str]:
        if not hasattr(self, "_collection"):
            logger.info("Loading the collection...")
            dataset = PintsAIDataset(
                self.global_cfg.dataset.pints_ai,
                self.global_cfg,
                self.collection_tokenizer,
            )
            dataset.post_processed_data = dataset.load_from_disk(
                dataset.post_process_cache_path
            )
            self._collection = dataset.post_processed_data
            logger.info("Collection loaded successfully.")
        return self._collection

    @property
    def index(self) -> faiss.Index:
        if not hasattr(self, "_index"):
            logger.info("Loading the index...")
            index = faiss.read_index(self.cfg.indexing.index_path)

            # Move the index to GPU if GPU is available
            if self.use_gpu:
                logger.info("Moving the index to the GPU...")
                # Create GPU resources
                res = faiss.StandardGpuResources()
                # Move the index to GPU
                index = faiss.index_cpu_to_gpu(res, 0, index)  # 0 is the GPU ID
                logger.info("Index moved to the GPU successfully.")

            # Set the index
            self._index = index

        return self._index

    @property
    def collection_tokenizer(self) -> AutoTokenizer:
        if not hasattr(self, "_collection_tokenizer"):
            self._collection_tokenizer = TOKENIZER_REGISTRY[
                self.global_cfg.model.name
            ].from_pretrained(self.global_cfg.model.base_name)
        return self._collection_tokenizer

    @property
    def index_tokenizer(self) -> AutoTokenizer:
        if not hasattr(self, "_index_tokenizer"):
            self._index_tokenizer = TOKENIZER_REGISTRY[
                self.global_cfg.model.name
            ].from_pretrained(self.cfg.encoding.model_name)
        return self._index_tokenizer

    @property
    def model(self) -> AutoModel:
        if not hasattr(self, "_model"):
            # Set the device map and attention implementation (use GPU if available. Config independent with index)
            device_map = "cuda:0" if torch.cuda.is_available() else None
            attn_impl = None if (torch.cuda.is_available() and is_torch_compile_possible()) else "eager"

            # Load the model
            self._model = AutoModel.from_pretrained(
                self.cfg.encoding.model_name,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
            if self.global_cfg.use_torch_compile and is_torch_compile_possible():
                logger.info("Compiling the model with torch compile...")
                self._model = torch.compile(self._model, dynamic=True)
            else:
                logger.info("Torch compile is not enabled.")
        return self._model

    def search(self, query: str, k: int, return_as_text: bool = False) -> List[Union[int, str]]:
        """
        Search for the top-k nearest neighbors of a given query string.
        """
        return self.search_batch([query], k, return_as_text)[0]

    def search_batch(self, queries: List[str], k: int, return_as_text: bool = False) -> List[List[Union[int, str]]]:
        """
        Search for the top-k nearest neighbors for a batch of queries.
        """
        query_embeddings = self.encode_queries(queries)
        _, indices = self.index.search(query_embeddings, k)
        indices = [lst.tolist() for lst in indices]
        if return_as_text:
            return [[self.convert_global_chunk_id_to_text(idx) for idx in lst] for lst in indices]
        return indices

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query into an embedding, using only the CLS token representation.
        """
        return self.encode_queries([query])[0]

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode a batch of queries into embeddings, using only the CLS token representation.
        """
        tokens = self.index_tokenizer(queries, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.inference_mode():
            outputs = self.model(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy() if torch.cuda.is_availabe() else outputs.last_hidden_state[:, 0].numpy()
        return cls_embeddings

    def convert_global_chunk_id_to_text(self, global_chunk_id: int) -> str:
        """
        Convert a global chunk ID to the corresponding text.
        """
        passage_id, local_chunk_id = self.convert_global_chunk_id_to_passage_id_and_local_chunk_id(global_chunk_id)
        start_idx = local_chunk_id * self.chunk_size
        end_idx = start_idx + self.chunk_size
        passage = self.collection[passage_id]["input_ids"]
        token_ids = passage[start_idx:end_idx]
        return self.collection_tokenizer.decode(token_ids, skip_special_tokens=True)

    def convert_global_chunk_id_to_passage_id_and_local_chunk_id(self, global_chunk_id: int) -> Tuple[int, int]:
        """
        Convert a global chunk ID to the corresponding passage ID and local chunk ID.
        """
        passage_id = global_chunk_id // self.num_chunks_per_passage
        local_chunk_id = global_chunk_id % self.num_chunks_per_passage
        return passage_id, local_chunk_id

    def convert_global_chunk_id_to_passage_id(self, global_chunk_id: int) -> int:
        """
        Convert a global chunk ID to the corresponding passage ID.
        """
        return global_chunk_id // self.num_chunks_per_passage

    def convert_global_chunk_id_to_local_chunk_id(self, global_chunk_id: int) -> int:
        """
        Convert a global chunk ID to the corresponding local chunk ID.
        """
        return global_chunk_id % self.num_chunks_per_passage