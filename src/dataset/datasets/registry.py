from typing import *

from .base_dataset import BaseDataset
from .LM.curation import CurationDataset
from .LM.lambada import LambadaDataset
from .LM.pints_ai import PintsAIDataset
from .LM.wikitext import WikiTextDataset
from .NLU.glue_cola import GLUECoLADataset
from .NLU.glue_sst2 import GLUESST2Dataset
from .NLU.glue_mrpc import GLUEMRPCDataset
from .NLU.glue_mnli import GLUEMNLIDataset
from .NLU.glue_qnli import GLUEQNLIDataset
from .NLU.glue_rte import GLUERTEDataset
from .NLU.glue_wnli import GLUEWNLIDataset
from .NLU.glue_qqp import GLUEQQPDataset
from .NLU.glue_stsb import GLUESTSBDataset

# Import SuperGLUE datasets
from .NLU.superglue_boolq import SuperGLUEBoolQDataset
from .NLU.superglue_cb import SuperGLUECBDataset
from .NLU.superglue_copa import SuperGLUECOPADataset
from .NLU.superglue_multirc import SuperGLUEMultiRCDataset
from .NLU.superglue_record import SuperGLUEReCoRDDataset
from .NLU.superglue_rte import SuperGLUERTEDataset
from .NLU.superglue_wic import SuperGLUEWiCDataset
from .NLU.superglue_wsc import SuperGLUEWSCDataset

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
    # Language modeling datasets
    "pints_ai": PintsAIDataset,
    "lambada": LambadaDataset,
    "wikitext": WikiTextDataset,
    "curation": CurationDataset,
    
    # GLUE datasets
    "glue_cola": GLUECoLADataset,
    "glue_sst2": GLUESST2Dataset,
    "glue_mrpc": GLUEMRPCDataset,
    "glue_mnli_matched": GLUEMNLIDataset,
    "glue_mnli_mismatched": GLUEMNLIDataset,
    "glue_qnli": GLUEQNLIDataset,
    "glue_rte": GLUERTEDataset,
    "glue_wnli": GLUEWNLIDataset,
    "glue_qqp": GLUEQQPDataset,
    "glue_stsb": GLUESTSBDataset,
    
    # SuperGLUE datasets
    "superglue_boolq": SuperGLUEBoolQDataset,
    "superglue_cb": SuperGLUECBDataset,
    "superglue_copa": SuperGLUECOPADataset,
    "superglue_multirc": SuperGLUEMultiRCDataset,
    "superglue_record": SuperGLUEReCoRDDataset,
    "superglue_rte": SuperGLUERTEDataset,
    "superglue_wic": SuperGLUEWiCDataset,
    "superglue_wsc": SuperGLUEWSCDataset,
}
