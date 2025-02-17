from typing import *

from .llama.model import Llama
from .rellama.model import ReLlama
from transformers import AutoModelForCausalLM

MODEL_REGISTRY: Dict[str, Type[Llama | ReLlama | AutoModelForCausalLM]] = {
    "llama": Llama,
    "rellama": ReLlama,
    "gpt": AutoModelForCausalLM,
}
