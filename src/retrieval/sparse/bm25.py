import logging
import os
from typing import *

import bm25s
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from src.retrieval.retriever import Retriever
from src.tokenization.registry import TOKENIZER_REGISTRY

logger = logging.getLogger("BM25Retriever")


class BM25Retriever(Retriever):
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        device: Optional[torch.device] = None,
        warmup: bool = False,
    ):
        super().__init__(cfg=cfg, global_cfg=global_cfg, device=device, warmup=warmup)

    @property
    def index_path(self) -> str:
        # TODO: Need to change the path to use multiple shards
        # TODO: Currently, we only use the first shard
        return os.path.join(
            self.cfg.dir_path,
            self.cfg.model.indexing.index_dir,
            f"shard_{self.cfg.model.indexing.total_shards}_0",
        )

    @property
    def index(self) -> bm25s.BM25:
        if not hasattr(self, "_index"):
            if not os.path.exists(self.index_path):
                logger.error(f"Index file not found at {self.index_path}")
                raise FileNotFoundError(f"Index file not found at {self.index_path}")
            logger.info(f"Loading index from {self.index_path}")
            self._index = bm25s.BM25.load(self.index_path)
        return self._index

    @index.setter
    def index(self, value: bm25s.BM25) -> None:
        if not isinstance(value, bm25s.BM25):
            raise TypeError("Index must be an instance of bm25s.BM25")
        self._index = value

    @property
    def tokenizer(self) -> AutoTokenizer:
        if not hasattr(self, "_index_tokenizer"):
            self._index_tokenizer = TOKENIZER_REGISTRY[
                self.global_cfg.model.name
            ].from_pretrained(self.cfg.model.encoding.model_name)
        return self._index_tokenizer

    def create_index(self, tokenized_corpus: Optional[List[List[int]]] = None) -> None:
        """
        Create the index for the retriever.
        """
        # Get the tokenized corpus
        if tokenized_corpus is None:
            tokenized_corpus = self.collection

        # Get vocabulary mapping
        vocab_mapping: Dict[str, int] = self.tokenizer.get_vocab()

        # Create the index
        self.index = bm25s.BM25()
        self.index.index((tokenized_corpus, vocab_mapping))

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Save the index
        self.index.save(self.index_path)

        return None

    def search_batch(
        self,
        queries: List[str],
        k: int,
        return_as_text: bool = False,
        passage_to_ignore_list: Optional[List[int]] = None,
        ensure_return_topk: bool = True,
    ) -> List[List[Union[int, str]]]:
        # Tokenize the queries
        # Remove the <bos> token
        tokenized_queries: List[List[int]] = [
            self.tokenizer(query)["input_ids"][1:] for query in queries
        ]

        return self.search_batch_with_tokens(
            tokens_batch=tokenized_queries,
            k=k,
            return_as_text=return_as_text,
            passage_to_ignore_list=passage_to_ignore_list,
            ensure_return_topk=ensure_return_topk,
        )

    def search_batch_with_tokens(
        self,
        tokens_batch: List[Union[torch.Tensor, List[int]]],
        k: int,
        return_as_text: bool = False,
        passage_to_ignore_list: Optional[List[int]] = None,
        ensure_return_topk: bool = True,
    ) -> List[List[Union[int, str]]]:
        # Save the original k
        original_k = k

        # Increase the k if there is passage to ignore
        if passage_to_ignore_list is not None:
            k = k + 1

        doc_ids, scores = self.index.retrieve(tokens_batch, k=k)

        # Ensure the number of results is correct
        if ensure_return_topk:
            avg_lens = sum(len(doc_ids[i]) for i in range(len(doc_ids))) / len(doc_ids)
            assert avg_lens == float(k), f"Expected {k} results, but got {avg_lens}"

        # Selected requested top k results
        doc_ids: List[List[int]] = [
            doc_ids[i][:original_k].tolist() for i in range(len(doc_ids))
        ]
        scores: List[List[float]] = [
            scores[i][:original_k].tolist() for i in range(len(scores))
        ]

        # Return the results
        if return_as_text:
            # Get the token ids from the corpus
            selected_token_ids_list: List[List[int]] = []
            selected_texts_list: List[List[str]] = []
            for bidx in range(len(doc_ids)):
                token_ids_list: List[List[int]] = []
                texts: List[str] = []
                for didx in range(len(doc_ids[bidx])):
                    token_ids_list.append(
                        self.collection[doc_ids[bidx][didx]]["input_ids"]
                    )
                    texts.append(
                        self.tokenizer.decode(
                            torch.tensor(token_ids_list[-1]), skip_special_tokens=True
                        )
                    )
                selected_token_ids_list.append(token_ids_list)
                selected_texts_list.append(texts)

            return selected_texts_list
        else:
            return doc_ids
