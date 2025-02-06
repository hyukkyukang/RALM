import logging
from typing import *

from transformers import PreTrainedTokenizerFast, AutoTokenizer
from src.tokenizer.utils import call_autotokenizer_with_hf_token

logger = logging.getLogger("RETROTokenizer")

class RETROTokenizer(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_name: str, hf_token: str = None, **kwargs):
        """Convert a pretrained tokenizer to a RETROTokenizer.

        :param model_name: _description_
        :type model_name: str
        :param hf_token: _description_, defaults to None
        :type hf_token: str, optional
        :return: _description_
        :rtype: _type_
        """
        tokenizer: AutoTokenizer = call_autotokenizer_with_hf_token(model_name=model_name, hf_token=hf_token, **kwargs)
        assert tokenizer.is_fast, "use_fast=True must be set for fast tokenization"
        return cls._initialize_tokenizer(tokenizer)

    @classmethod
    def _initialize_tokenizer(cls, tokenizer: AutoTokenizer) -> AutoTokenizer:
        # Set the last token as padding token if no padding token exists. 
        # We reuse the last token as padding token to avoid increasing the vocabulary size.
        if tokenizer.pad_token is None:
            last_token: str = tokenizer.convert_ids_to_tokens(len(tokenizer) - 1)
            logger.info(f"Setting the last token in the vocab ('{last_token}') as padding token")
            tokenizer.pad_token = last_token

        # Add dummy tokens if the number of tokens is not multiple of 64 (to increase throughput)
        # https://x.com/karpathy/status/1621578354024677377?lang=en for more details.
        if len(tokenizer) % 64 != 0:
            num_dummy_tokens: int = 64 - len(tokenizer) % 64
            new_total_token_num: int = len(tokenizer) + num_dummy_tokens
            logger.info(f"Adding {num_dummy_tokens} dummy tokens to make the number of tokens in the vocab to be a multiple of 64. (current: {len(tokenizer)}, new: {new_total_token_num})")
            dummy_tokens = [f'<dummy_token_{i+1}>' for i in range(num_dummy_tokens)]
            tokenizer.add_special_tokens({'additional_special_tokens': dummy_tokens})
        else:
            logger.info(f"The number of tokens in the vocab is already a multiple of 64. (current: {len(tokenizer)})")

        return tokenizer
