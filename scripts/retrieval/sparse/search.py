import logging
from typing import *

import hydra
from omegaconf import DictConfig

from src.dataset import DataModule
from src.retrieval.sparse.bm25 import BM25Retriever

logger = logging.getLogger("Search")


def test_custom_queries(retriever: BM25Retriever, return_as_text: bool = False) -> None:
    # Get the query
    # query = "What was one of Barack Obama’s major legislative accomplishments during his first term as President?"
    query = "Capture customer"

    # Get the retriever
    results: List[int] = retriever.search(query, return_as_text=return_as_text)

    # Print the results
    for idx, result in enumerate(results):
        logger.info(f"Result {idx}: {result}")
        with open(f"result2_{idx}.txt", "w") as f:
            f.write(f"{result}\n")

    return None


def test_training_corpus_queries(cfg: DictConfig, retriever: BM25Retriever) -> None:
    # Load the training dataset
    data_module = DataModule(cfg=cfg)
    data_module.prepare_data()

    chunk_size: int = 64

    # Get the retriever
    for idx, item in enumerate(data_module.train_dataset):
        if idx == 10:
            break
        num_chunks: int = len(item["input_ids"]) // chunk_size
        for chunk_idx in range(num_chunks):
            input_ids: List[int] = item["input_ids"][
                chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size
            ]
            input_texts: str = retriever.tokenizer.decode(
                input_ids, skip_special_tokens=True
            )
            retrived_doc_ids: List[int] = retriever.search_with_tokens(input_ids, 10)
            retrieved_doc_texts: List[str] = retriever.search_with_tokens(
                input_ids, 10, return_as_text=True
            )

            # Write to file
            with open(f"query_{idx}_{chunk_idx}.txt", "w") as f:
                f.write(f"Query: {input_texts}\n")
            with open(f"retrieved_doc_ids_{idx}_{chunk_idx}.txt", "w") as f:
                f.write(f"Retrieved doc ids: {retrived_doc_ids[0]}\n")
                f.write(f"Retrieved doc texts: {retrieved_doc_texts[0]}\n")
    return None


@hydra.main(
    version_base=None, config_path="/home/user/RALM/config", config_name="config"
)
def main(cfg: DictConfig) -> None:
    retriever = BM25Retriever(cfg.retrieval, cfg, device="cuda")

    test_custom_queries(retriever, return_as_text=True)
    # test_training_corpus_queries(cfg, retriever)
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
