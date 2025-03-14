import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import copy
import json
import logging
from functools import partial
from multiprocessing import Pool
from typing import *

import hkkang_utils.misc as misc_utils
import hydra
import tqdm
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

def shard_ids_to_global_chunk_ids(shard_ids: List[Tuple[int, int]]) -> List[int]:
    meta_path = "/root/RETRO/data/retrieval/chunk_ids/pints_ai/meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    num_items_per_shard: int = meta["num_items_per_shard"]["0"]
    global_chunk_ids: List[int] = []
    for shard_idx, shard_local_idx in shard_ids:
        global_chunk_ids.append(shard_idx * num_items_per_shard + shard_local_idx)
    return global_chunk_ids

def fix_retrieval_result(topk: int, global_chunk_id: int, dataset: BaseDataset, retrieved_chunk_dataset: RetrievedChunkDataset, retriever: SentenceTransformerRetriever, tokenizer: ReLlamaTokenizer) -> Tuple[int, int, List[int]]:
    # Convert the global chunk id to passage id and local chunk id
    passage_id, local_chunk_start_idx, local_chunk_end_idx = convert_global_chunk_id_to_passage_id_and_local_chunk_range(global_chunk_id, retriever.num_chunks_per_passage, retriever.chunk_size)
    
    # Get the query chunk for the retrieval
    query_chunk = dataset[passage_id]["input_ids"][local_chunk_start_idx:local_chunk_end_idx]
    query_str = tokenizer.decode(query_chunk, skip_special_tokens=True)

    # Perform the retrieval
    retrieved_global_chunk_ids: List[int] = retriever.search(query_str, k=topk, passage_id_to_ignore=passage_id)
    assert len(retrieved_global_chunk_ids) == topk, f"The number of retrieved global chunk ids is not equal to the topk: {len(retrieved_global_chunk_ids)} != {topk}"
    assert -1 not in retrieved_global_chunk_ids, f"The global chunk id {global_chunk_id} is not in the retrieved chunk ids"

    # Print the retrieved global chunk ids
    retrieved_texts: List[str] = [retriever.convert_global_chunk_id_to_text(global_chunk_id) for global_chunk_id in retrieved_global_chunk_ids]
    logger.info(f"Query: {query_str}")
    for i, retrieved_text in enumerate(retrieved_texts):
        logger.info(f"Retrieved {i+1}: {retrieved_text}")

    shard_idx, shard_local_idx = retrieved_chunk_dataset.global_chunk_id_to_shard_idx_and_local_idx(global_chunk_id)
    logger.info(f"Prepared modification for global chunk id: {global_chunk_id}")
    logger.info(f"Target shard idx: {shard_idx}, shard local idx: {shard_local_idx}")
    return (shard_idx, shard_local_idx, retrieved_global_chunk_ids)

def modify_shard_process(shard_idx, modifications_by_shard, retrieved_chunk_dataset):
    tmp = {shard_idx: modifications_by_shard[shard_idx]}
    retrieved_chunk_dataset.bulk_modify_retrieved_chunk_ids(tmp)
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

    with open("bad_shard_ids.json", "r") as f:
        shard_ids: List[Tuple[int, int]] = json.load(f)

    target_global_chunk_ids: List[int] = shard_ids_to_global_chunk_ids(shard_ids)
    modifications_by_shard = {}
    for idx, target_global_chunk_id in enumerate(tqdm.tqdm(target_global_chunk_ids, desc="Fixing retrieval result")):
        shard_idx, shard_local_idx, new_chunk_ids = fix_retrieval_result(
            topk=cfg.retrieval.topk,
            global_chunk_id=target_global_chunk_id,
            dataset=data_module.train_dataset,
            retrieved_chunk_dataset=retrieved_chunk_dataset,
            retriever=retriever,
            tokenizer=data_module.train_dataset.tokenizer
        )
        if shard_idx not in modifications_by_shard:
            modifications_by_shard[shard_idx] = {}
        modifications_by_shard[shard_idx][shard_local_idx] = new_chunk_ids

    # Apply bulk modifications for each shard in parallel using a process pool
    shard_indices = list(modifications_by_shard.keys())

    process_func = partial(modify_shard_process, modifications_by_shard=modifications_by_shard,
                           retrieved_chunk_dataset=retrieved_chunk_dataset)

    with Pool(processes=len(shard_indices)) as pool:
        results = list(tqdm.tqdm(pool.imap(process_func, shard_indices),
                                 total=len(shard_indices),
                                 desc="Modifying shards"))
    
    logger.info("Done!")



if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()