from typing import *

from .base_dataset import BaseDataset
from .LM.curation import CurationDataset
from .LM.lambada import LambadaDataset
from .LM.pints_ai import PintsAIDataset
from .LM.wikitext import WikiTextDataset
from .NLU.cola import CoLADataset
from .NLU.sst2 import SST2Dataset
from .NLU.mrpc import MRPCDataset

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
    "pints_ai": PintsAIDataset,
    "lambada": LambadaDataset,
    "wikitext": WikiTextDataset,
    "curation": CurationDataset,
    "cola": CoLADataset,
    "sst2": SST2Dataset,
    "mrpc": MRPCDataset,
}
