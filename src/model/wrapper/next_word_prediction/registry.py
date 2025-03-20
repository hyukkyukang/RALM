from typing import *

from .base import NextWordPredictor
from .gpt import NextWordPredictorForGPT
from .llama import NextWordPredictorForLlama
from .rellama import NextWordPredictorForReLlama


NEXT_WORD_PREDICTOR_REGISTRY: Dict[str, Type[NextWordPredictor]] = {
    "gpt": NextWordPredictorForGPT,
    "llama": NextWordPredictorForLlama,
    "rellama": NextWordPredictorForReLlama,
}
