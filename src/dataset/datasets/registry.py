from typing import *

from .base_dataset import BaseDataset
from .LM.curation import CurationDataset
from .LM.lambada import LambadaDataset
from .LM.pints_ai import PintsAIDataset
from .LM.wikitext import WikiTextDataset
from .NLU.cola import CoLADataset
from .NLU.sst2 import SST2Dataset
from .NLU.mrpc import MRPCDataset
from .NLU.mnli import MNLIDataset
from .NLU.qnli import QNLIDataset
from .NLU.rte import RTEDataset
from .NLU.wnli import WNLIDataset
from .NLU.qqp import QQPDataset

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
    "pints_ai": PintsAIDataset,
    "lambada": LambadaDataset,
    "wikitext": WikiTextDataset,
    "curation": CurationDataset,
    "cola": CoLADataset,
    "sst2": SST2Dataset,
    "mrpc": MRPCDataset,
    "mnli_matched": MNLIDataset,
    "mnli_mismatched": MNLIDataset,
    "qnli": QNLIDataset,
    "rte": RTEDataset,
    "wnli": WNLIDataset,
    "qqp": QQPDataset,
}
