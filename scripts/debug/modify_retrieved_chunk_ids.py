import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import copy
import logging
from typing import *

import hkkang_utils.misc as misc_utils
import hydra
from omegaconf import DictConfig

from src.dataset import DataModule
from src.dataset.datasets.base_dataset import BaseDataset
from src.retrieval.chunk_dataset import RetrievedChunkDataset
from src.retrieval.single_vector_retrieval.retriever import \
    SentenceTransformerRetriever
from src.retrieval.utils import \
    convert_global_chunk_id_to_passage_id_and_local_chunk_range
from src.tokenization import ReLlamaTokenizer

logger = logging.getLogger("FixRetrievalResult")

def fix_retrieval_result(topk: int, global_chunk_id: int, dataset: BaseDataset, retrieved_chunk_dataset: RetrievedChunkDataset, retriever: SentenceTransformerRetriever, tokenizer: ReLlamaTokenizer) -> None:
    # Convert the global chunk id to passage id and local chunk id
    passage_id, local_chunk_start_idx, local_chunk_end_idx = convert_global_chunk_id_to_passage_id_and_local_chunk_range(global_chunk_id, retriever.num_chunks_per_passage, retriever.chunk_size)
    # Get the query chunk for the retrieval
    query_chunk = dataset[passage_id]["input_ids"][local_chunk_start_idx:local_chunk_end_idx]
    query_str = tokenizer.decode(query_chunk, skip_special_tokens=True)
    # Perform the retrieval
    retrieved_global_chunk_ids: List[int] = retriever.search(query_str, k=topk, passage_id_to_ignore=passage_id)

    # Print the retrieved global chunk ids
    retrieved_texts: List[str] = [retriever.convert_global_chunk_id_to_text(global_chunk_id) for global_chunk_id in retrieved_global_chunk_ids]
    logger.info(f"Query: {query_str}")
    for i, retrieved_text in enumerate(retrieved_texts):
        logger.info(f"Retrieved {i+1}: {retrieved_text}")

    # Modify the retrieval chunk ids
    retrieved_chunk_dataset.modify_retrieved_chunk_ids(global_chunk_id, retrieved_global_chunk_ids)
    logger.info(f"Modified the retrieval chunk ids for the global chunk id: {global_chunk_id}")

    return None


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Initialize lightning module and call prepare_data to figure out the length of the dataset
    copy_cfg = copy.deepcopy(cfg)
    copy_cfg.model.is_use_retrieval = False
    data_module = DataModule(cfg=copy_cfg)
    data_module.prepare_data()
    
    # Get the retrieved chunk dataset to modify the retrieval chunk ids
    retrieved_chunk_dataset = RetrievedChunkDataset(dataset_name=data_module.train_dataset.name, cfg=cfg.retrieval, global_cfg=cfg)
    
    # Instantiate the retriever and move the model to the designated GPU.
    retriever = SentenceTransformerRetriever(cfg.retrieval, cfg)
    
    target_global_chunk_id = 1_556_342

    # Fix the retrieval result
    fix_retrieval_result(cfg.retrieval.topk, target_global_chunk_id, data_module.train_dataset, retrieved_chunk_dataset, retriever, tokenizer=data_module.train_dataset.tokenizer)
    
    logger.info("Done!")



if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
