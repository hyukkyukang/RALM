from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset


class SuperGLUECOPADataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(
        self, example: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Determine question type: cause or effect
        question_type = "cause" if example["question"] == 0 else "effect"

        context = (
            f"Read the following statement and determine the most plausible {question_type}.\n\n"
            f"Statement: {example['premise']}\n\n"
            f"Which of the following is the most likely {question_type}?\n"
            f"1. {example['choice1']}\n"
            f"2. {example['choice2']}\n\n"
            f"Provide your answer as '1' or '2'.\n"
            f"Answer:"
        )

        # The label is 0 or 1, indicating which choice is correct
        target = f"{example['label'] + 1}"  # Convert to "1" or "2"

        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["1", "2"],
            "idx": example.get("idx", 0),
        }
