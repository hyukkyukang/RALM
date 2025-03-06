from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset

class GLUEQQPDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        context = f"Are the following two questions semantically equivalent?\nQuestion1: {example['question1']}\nQuestion2: {example['question2']}\nAnswer:"
        # Set the target label
        if example["label"] == 0:
            target = "No"
        elif example["label"] == 1:
            target = "Yes"
        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["Yes", "No"],
        }