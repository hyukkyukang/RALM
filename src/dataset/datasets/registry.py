from typing import *

from .base_dataset import BaseDataset
from .LM.curation import CurationDataset
from .LM.lambada import LambadaDataset
from .LM.pints_ai import PintsAIDataset
from .LM.wikitext import WikiTextDataset
from .NLU.cola import CoLADataset

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
    "pints_ai": PintsAIDataset,
    "lambada": LambadaDataset,
    "wikitext": WikiTextDataset,
    "curation": CurationDataset,
    "cola": CoLADataset,
}
