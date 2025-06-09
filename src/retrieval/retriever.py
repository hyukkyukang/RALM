import abc
import logging
from typing import *

import faiss
import hkkang_utils.pattern as pattern_utils
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from src.retrieval.utils import (
    convert_global_chunk_id_to_passage_id,
    convert_global_chunk_id_to_passage_id_and_local_chunk_range,
)
from src.tokenization.registry import TOKENIZER_REGISTRY

logger = logging.getLogger("Retriever")


class Retriever(metaclass=pattern_utils.SingletonMetaWithArgs):
    """
    Abstract base class for retrieval systems.

    This class defines the interface for different retrieval implementations,
    providing common functionality for searching and processing text collections.
    It uses the SingletonMetaWithArgs pattern to ensure only one instance exists
    for a given configuration.

    Attributes:
        cfg (DictConfig): Configuration specific to the retriever implementation.
        global_cfg (DictConfig): Global configuration shared across the application.
        device (Optional[torch.device]): Device to use for computations (CPU/GPU).
    """

    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        device: Optional[torch.device] = None,
        warmup: bool = False,
    ) -> None:
        """
        Initialize the Retriever with configuration and device settings.

        Args:
            cfg (DictConfig): Configuration specific to the retriever implementation.
            global_cfg (DictConfig): Global configuration shared across the application.
            device (Optional[torch.device]): Device to use for computations (CPU/GPU).
        """
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.device = device
        self.warmup = warmup

        if warmup:
            self.warmup_index()

    @property
    @abc.abstractmethod
    def index(self) -> Union[faiss.Index, None]:
        """
        Abstract property for the index used for text similarity search, whether in sparse or dense retrieval.

        Returns:
            Union[faiss.Index, None]: The index for similarity search.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Index is not implemented for this dataset.")

    @property
    @abc.abstractmethod
    def tokenizer(self) -> AutoTokenizer:
        """
        Abstract property for the tokenizer used to process text.

        Returns:
            AutoTokenizer: The tokenizer for text processing.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Tokenizer is not implemented for this dataset.")

    @property
    def use_gpu(self) -> bool:
        """
        Determine if GPU should be used for computations.

        Returns:
            bool: True if GPU is available and configured to be used, False otherwise.
        """
        return self.cfg.use_gpu and torch.cuda.is_available()

    @property
    def chunk_size(self) -> int:
        """
        Get the size of text chunks for processing.

        Returns:
            int: The size of text chunks as defined in the global configuration.
        """
        return self.global_cfg.model.input_chunk_size

    @property
    def num_chunks_per_passage(self) -> int:
        """
        Calculate the number of chunks per passage based on model configuration.

        Returns:
            int: The number of chunks that can fit in a passage based on model max length.
        """
        return self.global_cfg.model.max_length // self.chunk_size

    @property
    def collection(self) -> List[Dict[str, Any]]:
        """
        Lazy-load and return the text collection for retrieval.

        This property loads the collection from disk if it hasn't been loaded yet,
        using the dataset registry to get the appropriate dataset implementation.

        Returns:
            List[Dict[str, Any]]: The loaded text collection with processed data.
        """
        if not hasattr(self, "_collection"):
            corpus_name = self.cfg.corpus_name
            logger.info(f"Loading the collection for {corpus_name}...")
            from src.dataset import DATASET_REGISTRY

            dataset = DATASET_REGISTRY[corpus_name](
                self.global_cfg.dataset[corpus_name],
                self.global_cfg,
                self.collection_tokenizer,
            )
            dataset.post_processed_data = dataset.load_from_disk(
                dataset.post_process_cache_path
            )
            self._collection = dataset.post_processed_data
            logger.info(f"Collection for {corpus_name} loaded successfully.")
        return self._collection

    @property
    def collection_tokenizer(self) -> AutoTokenizer:
        """
        Lazy-load and return the tokenizer for the collection.

        Returns:
            AutoTokenizer: The tokenizer for processing the collection text.
        """
        if not hasattr(self, "_collection_tokenizer"):
            self._collection_tokenizer = TOKENIZER_REGISTRY[
                self.global_cfg.model.name
            ].from_pretrained(self.global_cfg.model.base_name)
        return self._collection_tokenizer

    @abc.abstractmethod
    def create_index(self, *args, **kwargs) -> None:
        """
        Create the index for the retriever.
        """
        raise NotImplementedError("Index creation is not implemented for this dataset.")

    @abc.abstractmethod
    def search_batch(
        self,
        queries: List[str],
        k: int,
        return_as_text: bool = False,
        passage_to_ignore_list: Optional[List[int]] = None,
        ensure_return_topk: bool = True,
    ) -> List[List[Union[int, str]]]:
        """
        Search for the top-k nearest neighbors for a batch of queries.

        Args:
            queries (List[str]): List of query strings to search for.
            k (int): Number of nearest neighbors to retrieve for each query.
            return_as_text (bool, optional): Whether to return results as text instead of IDs. Defaults to False.
            passage_to_ignore_list (Optional[List[int]], optional): List of passage IDs to exclude from results. Defaults to None.
            ensure_return_topk (bool, optional): Whether to ensure exactly k results are returned. Defaults to True.

        Returns:
            List[List[Union[int, str]]]: For each query, a list of k nearest neighbor IDs or text strings.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Search batch is not implemented for this dataset.")

    @abc.abstractmethod
    def search_batch_with_tokens(
        self,
        tokens_batch: List[Union[torch.Tensor, List[int]]],
        k: int,
        return_as_text: bool = False,
        passage_to_ignore_list: Optional[List[int]] = None,
        ensure_return_topk: bool = True,
    ) -> List[List[Union[int, str]]]:
        """
        Search for the top-k nearest neighbors for a batch of tokenized queries.

        Args:
            tokens_batch (Dict[str, torch.Tensor]): Batch of tokenized queries with keys like 'input_ids' and 'attention_mask'.
            k (int): Number of nearest neighbors to retrieve for each query.
            return_as_text (bool, optional): Whether to return results as text instead of IDs. Defaults to False.
            passage_to_ignore_list (Optional[List[int]], optional): List of passage IDs to exclude from results. Defaults to None.
            ensure_return_topk (bool, optional): Whether to ensure exactly k results are returned. Defaults to True.

        Returns:
            List[List[Union[int, str]]]: For each query, a list of k nearest neighbor IDs or text strings.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Search batch with tokens is not implemented for this dataset."
        )

    def warmup_index(self) -> None:
        """
        Warmup the index for the retriever.
        """
        logger.info("Warmup the index...")
        self.index
        logger.info("Warmup the collection...")
        self.collection
        logger.info("Warmup completed.")

    def search(
        self,
        query: str,
        k: int = 10,
        return_as_text: bool = False,
        passage_id_to_ignore: Optional[int] = None,
        ensure_return_topk: bool = True,
    ) -> List[Union[int, str]]:
        """
        Search for the top-k nearest neighbors of a given query string.

        This is a convenience method that wraps search_batch for a single query.

        Args:
            query (str): The query string to search for.
            k (int): Number of nearest neighbors to retrieve.
            return_as_text (bool, optional): Whether to return results as text instead of IDs. Defaults to False.
            passage_id_to_ignore (Optional[int], optional): Passage ID to exclude from results. Defaults to None.
            ensure_return_topk (bool, optional): Whether to ensure exactly k results are returned. Defaults to True.

        Returns:
            List[Union[int, str]]: List of k nearest neighbor IDs or text strings.
        """
        passage_to_ignore_list: Optional[List[int]] = (
            [passage_id_to_ignore] if passage_id_to_ignore is not None else None
        )
        return self.search_batch(
            [query],
            k,
            return_as_text,
            passage_to_ignore_list=passage_to_ignore_list,
            ensure_return_topk=ensure_return_topk,
        )[0]

    def search_with_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        k: int,
        return_as_text: bool = False,
        passage_to_ignore_list: Optional[List[int]] = None,
    ) -> List[Union[int, str]]:
        """
        Search for the top-k nearest neighbors for a single tokenized query.

        This is a convenience method that wraps search_batch_with_tokens for a single tokenized query.

        Args:
            tokens (Dict[str, torch.Tensor]): Tokenized query with keys like 'input_ids' and 'attention_mask'.
            k (int): Number of nearest neighbors to retrieve.
            return_as_text (bool, optional): Whether to return results as text instead of IDs. Defaults to False.
            passage_to_ignore_list (Optional[List[int]], optional): List of passage IDs to exclude from results. Defaults to None.

        Returns:
            List[Union[int, str]]: List of k nearest neighbor IDs or text strings.
        """
        return self.search_batch_with_tokens(
            [tokens], k, return_as_text, passage_to_ignore_list
        )[0]

    def convert_global_chunk_id_to_text(self, global_chunk_id: int) -> str:
        """
        Convert a global chunk ID to the corresponding text.

        Args:
            global_chunk_id (int): The global chunk ID to convert.

        Returns:
            str: The text content of the chunk.
        """
        passage_id, chunk_start_idx, chunk_end_idx = (
            convert_global_chunk_id_to_passage_id_and_local_chunk_range(
                global_chunk_id, self.num_chunks_per_passage, self.chunk_size
            )
        )
        passage = self.collection[passage_id]["input_ids"]
        token_ids = passage[chunk_start_idx:chunk_end_idx]
        return self.collection_tokenizer.decode(token_ids, skip_special_tokens=True)

    def _filter_by_passage_id(
        self, global_chunk_ids: List[int], passage_id_to_ignore: int
    ) -> List[int]:
        """
        Filter the global chunk IDs by the passage ID to ignore.

        Args:
            global_chunk_ids (List[int]): List of global chunk IDs to filter.
            passage_id_to_ignore (int): Passage ID to exclude from results.

        Returns:
            List[int]: Filtered list of global chunk IDs, excluding those from the specified passage.
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
