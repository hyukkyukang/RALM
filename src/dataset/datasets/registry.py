from typing import *

from .base_dataset import BaseDataset
from .curation import CurationDataset
from .lambada import LambadaDataset
from .pints_ai import PintsAIDataset
from .wikitext import WikiTextDataset

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
    "pints_ai": PintsAIDataset,
    "lambada": LambadaDataset,
    "wikitext": WikiTextDataset,
    "curation": CurationDataset,
}
