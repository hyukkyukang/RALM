import logging
from typing import *

import hkkang_utils.misc as misc_utils
import hydra
from omegaconf import DictConfig

from src.dataset import PintsAIDataset
from src.retrieval import SentenceTransformerRetriever
from src.tokenization import TOKENIZER_REGISTRY

logger = logging.getLogger("Search")


def search_with_user_query(retriever: SentenceTransformerRetriever) -> None:
    queries = ["What is the capital of France?", "What is the capital of Germany?"]
    k = 5
    results: List[Union[str, int]] = retriever.search_batch(
        queries=queries, k=k, return_as_text=True
    )
    logging.info(f"Results with user query: {results}")
    return None


def search_with_chunks_from_dataset(
    retriever: SentenceTransformerRetriever,
    global_cfg: DictConfig,
    k: int = 5,
    test_num: int = 10,
) -> None:
    # Load the collection
    tokenizer = TOKENIZER_REGISTRY[global_cfg.model.name].from_pretrained(
        global_cfg.model.base_name
    )
    dataset = PintsAIDataset(
        global_cfg.dataset.pints_ai,
        global_cfg,
        tokenizer,
    )
    dataset.post_processed_data = dataset.load_from_disk(
        dataset.post_process_cache_path
    )

    def get_chunk_from_dataset():
        for idx, item in enumerate(dataset):
            token_ids = item["input_ids"]
            assert len(token_ids) % retriever.chunk_size == 0
            chunk_id = 0
            for idx in range(0, len(token_ids), retriever.chunk_size):
                chunk = token_ids[idx : idx + retriever.chunk_size]
                yield chunk, chunk_id
                chunk_id += 1

    # Get the first chunk
    iter_get_chunk_from_dataset = get_chunk_from_dataset()
    for _ in range(test_num):
        chunk, global_chunk_id = next(iter_get_chunk_from_dataset)
        passage_id = retriever.convert_global_chunk_id_to_passage_id(global_chunk_id)
        # Get the text
        text = tokenizer.decode(chunk, skip_special_tokens=True)
        # Search for the top-k nearest neighbors
        queries = [text]
        # Search with ignore list
        results_with_ignore_list: List[Union[str, int]] = retriever.search_batch(
            queries=queries,
            k=k,
            return_as_text=True,
            passage_to_ignore_list=[passage_id],
        )
        # Search without ignore list
        results_without_ignore_list: List[Union[str, int]] = retriever.search_batch(
            queries=queries, k=k, return_as_text=True
        )
        # Print the results
        logging.info(f"Results with ignore list: {results_with_ignore_list}")
        logging.info(f"Results without ignore list: {results_without_ignore_list}")

    return None


@hydra.main(version_base=None, config_path="/home/user/RALM/config", config_name="config")
def main(cfg: DictConfig) -> None:
    retriever = SentenceTransformerRetriever(cfg=cfg.retrieval, global_cfg=cfg)

    search_with_user_query(retriever)

    search_with_chunks_from_dataset(retriever, cfg)

    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
