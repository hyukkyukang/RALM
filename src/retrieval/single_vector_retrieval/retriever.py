import logging
import os
from typing import Dict, List, Optional, Union

import faiss
import numpy as np
import torch
from datasets import Dataset
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.retrieval.dataloader import StreamingCorpusDataLoader
from src.retrieval.utils import (
    convert_global_chunk_id_to_passage_id,
    convert_global_chunk_id_to_passage_id_and_local_chunk_range,
)
from src.tokenization.registry import TOKENIZER_REGISTRY
from src.utils import AsyncChunkIDSaver, is_main_process, is_torch_compile_possible

logger = logging.getLogger("SentenceTransformerRetriever")


class SentenceTransformerCorpusRetriever:
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        save_dir_path: str,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.retriever = SentenceTransformerRetriever(cfg, global_cfg, device)
        self.save_dir_path = save_dir_path
        self.__post_init__()

    def __post_init__(self):
        if not os.path.exists(self.save_dir_path):
            os.makedirs(self.save_dir_path)

    @property
    def dummy_retrieved_chunk_ids(self) -> List[List[int]]:
        return [-1] * self.cfg.topk

    def retrieve_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        data_span_start_idx: int,
        data_span_end_idx: int,
        num_dataloader_workers: int = 4,
    ) -> None:
        # Create a DataLoader that streams data from the dataset.
        dataloader = StreamingCorpusDataLoader(
            dataset,
            start_idx=data_span_start_idx,
            end_idx=data_span_end_idx,
            src_tokenizer=self.retriever.collection_tokenizer,
            tgt_tokenizer=self.retriever.index_tokenizer,
            chunk_size=self.retriever.chunk_size,
            batch_size=batch_size,
            num_workers=num_dataloader_workers,
        )
        logger.info(
            f"Retrieving dataset from {data_span_start_idx} to {data_span_end_idx}"
        )
        # Instantiate AsyncSaver to perform disk writes asynchronously.
        async_saver = AsyncChunkIDSaver(max_queue_size=3)

        disable_tqdm = not is_main_process()
        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                desc="Retrieving dataset",
                total=len(dataloader),
                disable=disable_tqdm,
            )
        ):
            num_passages: int = len(batch["num_chunks"])
            # Calculate the starting index for the current batch.
            min_passage_idx = min(batch["passage_indices"])
            # Build the file path for this batch.
            file_path = os.path.join(
                self.save_dir_path,
                f"chunk_ids_{min_passage_idx}_{num_passages}.npy",
            )

            # Skip batch if file already exists.
            if os.path.exists(file_path):
                continue

            # Move the batch to the model device.
            batch_input = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            passage_indices = [item for item in batch["passage_indices"]]

            # Perform the retrieval.
            chunk_ids: List[List[int]] = self.retriever.search_batch_with_tokens(
                batch_input,
                k=self.cfg.topk,
                passage_to_ignore_list=passage_indices,
            )

            # Add dummy chunk_ids if the number of chunks is less than the maximum number of chunks per passage.
            new_chunk_ids: List[List[int]] = []

            # Check if the number of chunks per passage is greater than the maximum number of chunks per passage.
            # We need this condition for the code to work.
            assert (
                max(batch["num_chunks"]) <= self.retriever.num_chunks_per_passage
            ), "The number of chunks per passage is greater than the maximum number of chunks per passage."

            # Add the retrieved chunk_ids and dummy chunk_ids.
            cnt = 0
            for num_chunks in batch["num_chunks"]:
                # Add the retrieved chunk_ids
                new_chunk_ids.extend(chunk_ids[cnt : cnt + num_chunks])
                cnt += num_chunks

                # Add dummy chunk_ids if the number of chunks is less than the maximum number of chunks per passage.
                if num_chunks < self.retriever.num_chunks_per_passage:
                    for _ in range(self.retriever.num_chunks_per_passage - num_chunks):
                        new_chunk_ids.append(self.dummy_retrieved_chunk_ids)

            # Replace the old chunk_ids with the new chunk_ids
            chunk_ids = new_chunk_ids

            # Enqueue saving the embeddings asynchronously.
            async_saver.save(file_path, chunk_ids)

        # Wait for all asynchronous saves to complete.
        async_saver.close()
        return None


class SentenceTransformerRetriever:
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.device = device

    @property
    def use_gpu(self) -> bool:
        return self.cfg.use_gpu and torch.cuda.is_available()

    @property
    def chunk_size(self) -> int:
        return self.global_cfg.model.input_chunk_size

    @property
    def num_chunks_per_passage(self) -> int:
        return self.global_cfg.model.max_length // self.chunk_size

    @property
    def collection(self) -> List[str]:
        if not hasattr(self, "_collection"):
            logger.info("Loading the collection...")
            from src.dataset import DATASET_REGISTRY

            dataset = DATASET_REGISTRY[self.cfg.corpus_name](
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
            index_path = os.path.join(
                self.global_cfg.root_dir_path,
                self.cfg.dir_path,
                self.cfg.indexing.index_dir,
                f"{self.cfg.corpus_name}_{self.cfg.indexing.index_file_name}",
            )
            index = faiss.read_index(index_path)

            # Move the index to GPU if GPU is available
            if self.use_gpu:
                logger.info("Moving the index to the GPU...")
                # Create GPU resources
                res = faiss.StandardGpuResources()
                # Get the device id
                # Expects device to be set by torch.cuda.set_device(device) when doing multi-GPU training
                device_id = torch.cuda.current_device()
                # Move the index to GPU
                index = faiss.index_cpu_to_gpu(res, device_id, index)
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
            # Expects device to be set by torch.cuda.set_device(device) when doing multi-GPU training
            device_map = (
                torch.device(f"cuda:{torch.cuda.current_device()}")
                if torch.cuda.is_available()
                else None
            )
            attn_impl = (
                None
                if (torch.cuda.is_available() and is_torch_compile_possible())
                else "eager"
            )

            # Load the model
            self._model = AutoModel.from_pretrained(
                self.cfg.encoding.model_name,
                device_map=device_map,
                attn_implementation="eager",
            )
            if self.global_cfg.use_torch_compile and is_torch_compile_possible():
                logger.info("Compiling the model with torch compile...")
                self._model = torch.compile(self._model, dynamic=True)
            else:
                logger.info("Torch compile is not enabled.")
        return self._model

    def search(
        self,
        query: str,
        k: int,
        return_as_text: bool = False,
        passage_id_to_ignore: int = None,
        ensure_return_topk: bool = True,
    ) -> List[Union[int, str]]:
        """
        Search for the top-k nearest neighbors of a given query string.
        """
        return self.search_batch(
            [query], k, return_as_text, passage_to_ignore_list=[passage_id_to_ignore], ensure_return_topk=ensure_return_topk
        )[0]

    def search_batch(
        self,
        queries: List[str],
        k: int,
        return_as_text: bool = False,
        passage_to_ignore_list: List[int] = None,
        ensure_return_topk: bool = True,
    ) -> List[List[Union[int, str]]]:
        """
        Search for the top-k nearest neighbors for a batch of queries.
        """
        # Encode the queries
        query_embeddings = self.encode_queries(queries)

        return self.search_batch_with_embeddings(
            query_embeddings, k, return_as_text, passage_to_ignore_list, ensure_return_topk
        )

    def search_with_tokens(
        self,
        tokens: Dict[str, torch.Tensor],
        k: int,
        return_as_text: bool = False,
        passage_to_ignore_list: List[int] = None,
    ) -> List[List[Union[int, str]]]:
        """
        Search for the top-k nearest neighbors for a single token.
        """
        return self.search_batch_with_tokens(
            [tokens], k, return_as_text, passage_to_ignore_list
        )[0]

    def search_batch_with_tokens(
        self,
        tokens_batch: Dict[str, torch.Tensor],
        k: int,
        return_as_text: bool = False,
        passage_to_ignore_list: List[int] = None,
        ensure_return_topk: bool = True,
    ) -> List[List[Union[int, str]]]:
        """
        Search for the top-k nearest neighbors for a batch of tokens.
        """
        query_embeddings = self.encode_tokens_batch(tokens_batch)
        return self.search_batch_with_embeddings(
            query_embeddings, k, return_as_text, passage_to_ignore_list, ensure_return_topk
        )

    def search_batch_with_embeddings(
        self,
        query_embeddings: np.ndarray,
        k: int,
        return_as_text: bool = False,
        passage_to_ignore_list: List[int] = None,
        ensure_return_topk: bool = True,
    ) -> List[List[Union[int, str]]]:
        """
        Search for the top-k nearest neighbors for a batch of queries.
        """
        # Increase the number of k if there are passages to ignore
        original_k: int = k
        if passage_to_ignore_list is not None:
            k = k + self.num_chunks_per_passage

        # Search for the top-k nearest neighbors
        _, indices = self.index.search(query_embeddings, k)
        indices: List[List[int]] = [lst.tolist() for lst in indices]

        # Remove the passages to ignore
        if passage_to_ignore_list is not None:
            indices = [self._filter_by_passage_id(lst, passage_to_ignore_list[b_idx]) for b_idx, lst in enumerate(indices)]

        # Get the top-k nearest neighbors
        indices = [lst[:original_k] for lst in indices]

        if ensure_return_topk:
            # Count valid indices
            valid_indices = sum(i != -1 for lst in indices for i in lst)
            # Search with increased nprobe if the number of valid indices is less than the original k
            if valid_indices < original_k:
                # Search with increased nprobe
                original_nprobe = self.index.nprobe
                self.index.nprobe = k*2
                _, indices = self.index.search(query_embeddings, k)
                indices: List[List[int]] = [lst.tolist() for lst in indices]

                # Remove the passages to ignore
                if passage_to_ignore_list is not None:
                    indices = [self._filter_by_passage_id(lst, passage_to_ignore_list[b_idx]) for b_idx, lst in enumerate(indices)]

                # Get the top-k nearest neighbors
                indices = [lst[:original_k] for lst in indices]

                # Restore the original nprobe
                self.index.nprobe = original_nprobe

            # Count valid indices
            valid_indices = sum(i != -1 for lst in indices for i in lst)

            assert valid_indices == original_k, "The number of valid indices is not equal to the original k"

        # Convert the indices to text if requested
        if return_as_text:
            return [
                [self.convert_global_chunk_id_to_text(idx) for idx in lst]
                for lst in indices
            ]

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
        return self.encode_tokens_batch(tokens)

    def encode_tokens(self, tokens: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Encode a batch of tokens into embeddings, using only the CLS token representation.
        """
        tokens_batch = {k: v.unsqueeze(0) for k, v in tokens.items()}
        return self.encode_tokens_batch(tokens_batch)[0]

    def encode_tokens_batch(self, tokens_batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Encode a batch of tokens into embeddings, using only the CLS token representation.
        """
        device = next(self.model.parameters()).device
        tokens_batch = {k: v.to(device) for k, v in tokens_batch.items()}
        with torch.inference_mode():
            outputs = self.model(**tokens_batch)
            cls_embeddings = (
                outputs.last_hidden_state[:, 0].cpu().numpy()
                if torch.cuda.is_available()
                else outputs.last_hidden_state[:, 0].numpy()
            )
        return cls_embeddings

    def convert_global_chunk_id_to_text(self, global_chunk_id: int) -> str:
        """
        Convert a global chunk ID to the corresponding text.
        """
        passage_id, chunk_start_idx, chunk_end_idx = (
            convert_global_chunk_id_to_passage_id_and_local_chunk_range(
                global_chunk_id, self.num_chunks_per_passage, self.chunk_size
            )
        )
        passage = self.collection[passage_id]["input_ids"]
        token_ids = passage[chunk_start_idx:chunk_end_idx]
        return self.collection_tokenizer.decode(token_ids, skip_special_tokens=True)


    def _filter_by_passage_id(self, global_chunk_ids: List[int], passage_id_to_ignore: int) -> List[int]:
        """
        Filter the global chunk IDs by the passage ID to ignore.
        """
        filtered_global_chunk_ids: List[int] = []
        for global_chunk_id in global_chunk_ids:
            # Convert the global chunk ID to the passage ID
            passage_id: int = convert_global_chunk_id_to_passage_id(
                global_chunk_id, self.num_chunks_per_passage
            )
            if passage_id != passage_id_to_ignore:
                filtered_global_chunk_ids.append(global_chunk_id)
        return filtered_global_chunk_ids