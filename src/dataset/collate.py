from transformers import DataCollatorForLanguageModeling
from typing import Dict, List, Any


class ReLLamaDataCollator(DataCollatorForLanguageModeling):
    """A custom data collator for ReLLama that extends the HuggingFace DataCollatorForLanguageModeling.

    This collator adds additional functionality to:
        1. Track character counts for each sequence
    """

    def __init__(self, tokenizer: Any, mlm: bool = False) -> None:
        """Initialize the ReLLama data collator.

        Args:
            tokenizer: The tokenizer to use for encoding/decoding text
            mlm: Whether to use masked language modeling (default: False)
        """
        super().__init__(tokenizer=tokenizer, mlm=mlm)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of examples. Add character counts for each sequence, after collation by the parent class.

        Args:
            examples: List of examples containing input_ids and other fields

        Returns:
            Dict containing:
                - All fields from parent collator (input_ids, labels, etc.)
                - char_counts: List of character counts for each sequence
        """
        # First apply the parent class collation
        batch = super().__call__(examples)

        # Decode individual tokens and count characters
        char_counts: List[int] = []
        avg_char_in_token: List[float] = []
        for b_idx, token_ids in enumerate(batch["input_ids"]):
            # Count characters in the full text (excluding special tokens)
            full_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            char_counts.append(len(full_text))  # Count characters directly
            avg_char_in_token.append(
                len(full_text) / sum(batch["attention_mask"][b_idx])
            )

        # Compute the average number of characters per token (in the batch)
        avg_char_in_token = sum(avg_char_in_token) / len(avg_char_in_token)

        batch["char_counts"] = char_counts
        batch["avg_char_in_token"] = avg_char_in_token

        return batch
