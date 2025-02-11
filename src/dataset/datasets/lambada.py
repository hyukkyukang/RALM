import logging

from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from src.dataset.datasets.base_dataset import BaseDataset
from src.dataset.utils import split_text_into_context_and_last_word
from src.utils import log_if_rank_zero

logger = logging.getLogger("LambadaDataset")


class LambadaDataset(BaseDataset):
    def __init__(self, cfg: DictConfig, tokenized_data: Dataset | None = None):
        super().__init__(cfg, tokenized_data)

    def _load_dataset(self) -> Dataset:
        dataset = load_dataset(
            self.cfg.dataset.huggingface_dataset_name,
            split=self.cfg.dataset.split,
            cache_dir=self.hf_cache_dir_path,
            num_proc=8,
        )
        log_if_rank_zero(
            logger,
            f"Splitting {len(dataset)} text into context and last word...",
        )
        dataset = dataset.map(
            lambda x: split_text_into_context_and_last_word(x["text"]),
            batched=False,
        )
        return dataset
