from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from src.dataset.datasets.base_dataset import BaseDataset


class WikiTextDataset(BaseDataset):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _load_dataset(self) -> Dataset:
        return load_dataset(
            self.cfg.dataset.huggingface_dataset_name,
            split=self.cfg.dataset.split,
            cache_dir=self.hf_cache_dir_path,
            num_proc=8,
        )
