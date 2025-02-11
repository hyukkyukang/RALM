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
