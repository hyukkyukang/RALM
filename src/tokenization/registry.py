from typing import *

from .llama_tokenizer import LlamaTokenizer
from .rellama_tokenizer import ReLlamaTokenizer
from transformers import AutoTokenizer

TOKENIZER_REGISTRY: Dict[
    str, Type[LlamaTokenizer | ReLlamaTokenizer | AutoTokenizer]
] = {
    "llama": LlamaTokenizer,
    "rellama": ReLlamaTokenizer,
    "gpt": AutoTokenizer,
}
