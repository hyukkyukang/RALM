from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset

class SuperGLUEBoolQDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        context = (
        f"Read the following passage carefully:\n\n"
        f"Passage: {example['passage']}\n\n"
        f"Based on the passage, answer the following yes/no question.\n\n"
        f"Question: {example['question']}\n\n"
        f"Your answer should be either 'Yes' or 'No'.\n"
        f"Answer:"
        )

        target = "Yes" if example["label"] == 1 else "No"

        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["Yes", "No"],
        }