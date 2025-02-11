import logging
from typing import *

from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from src.dataset.datasets.base_dataset import BaseDataset
from src.dataset.utils import split_text_into_context_and_last_word
from src.utils import log_if_rank_zero

logger = logging.getLogger("LambadaDataset")


class LambadaDataset(BaseDataset):
    """Dataset class for the LAMBADA (Language Modeling Broadened to Account for Discourse Aspects) dataset.
    This dataset is designed to evaluate the capability of models to understand long-range dependencies."""
    
    def __init__(self, cfg: DictConfig, tokenized_data: Dataset | None = None):
        super().__init__(cfg, tokenized_data)

    def _load_dataset(self) -> Dataset:
        """Loads and preprocesses the LAMBADA dataset.
        
        Returns:
            Dataset: Processed dataset with text split into context and last word.
        """
        # Load the raw dataset from Hugging Face
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
        # Process each example by splitting text into context and last word
        dataset = dataset.map(
            lambda x: split_text_into_context_and_last_word(x["text"]),
            batched=False,
        )
        return dataset


class LambadaDataCollator:
    """Collator class for LAMBADA dataset that prepares data for model input.
    Note: Currently does not support batch processing."""
    
    def __init__(self, tokenizer: Any, mlm: bool = False) -> None:
        """Initialize the collator.
        
        Args:
            tokenizer: Tokenizer to be used for processing text
            mlm: Flag for masked language modeling (currently unused)
        """
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collates a list of examples, but we do not make batched tensor.
        Currently, lambada evaluation code does not support batch processing anyway.
        
        Args:
            examples: List of dictionaries containing the dataset examples
            
        Returns:
            Dict containing batched data with same keys as input examples
        """
        # Get keys of the first example
        first_example_keys = examples[0].keys()
        # Create a dictionary to store the batch data
        batch = {key: [] for key in first_example_keys}
        
        # Iterate over the examples and add the data to the batch
        for example in examples:
            for key in first_example_keys:
                batch[key].append(example[key])

        return batch
