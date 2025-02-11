import re
from typing import *

import hkkang_utils.pattern as pattern_utils
from transformers import BasicTokenizer


@pattern_utils.singleton
class SingletonBasicTokenizer:
    def __init__(self):
        self.basic_tokenizer = BasicTokenizer()

    def tokenize(self, text: str) -> List[str]:
        return self.basic_tokenizer.tokenize(text)


def split_text_into_context_and_last_word(line: str) -> Dict[str, str]:
    line = line.strip()
    basic_tokenizer = SingletonBasicTokenizer()
    toks = basic_tokenizer.tokenize(line)
    length_of_word = len(toks[-1])
    assert length_of_word > 0, f"The last word is empty: {toks[-1]}"
    return {"context": line[:-length_of_word].strip(), "last_word": toks[-1]}

def normalize_quotes(text: str) -> str:
    """
    Normalize various types of single and double quotes to standard quotes (' and ").
    Handles curly quotes, prime marks, and their combinations.
    Uses regex for efficient pattern matching.
    """
    # Map of quote patterns to be replaced with standard quotes
    quote_patterns = [
        (r'[""‟„″]', '"'),  # various double quotes to standard double quote
        (r"``", '"'),  # double backticks to double quote
        (r"[" "‛′]", "'"),  # various single quotes to standard single quote
        (r"`", "'"),  # single backtick to single quote
    ]

    result = text
    for pattern, replacement in quote_patterns:
        result = re.sub(pattern, replacement, result)

    return result