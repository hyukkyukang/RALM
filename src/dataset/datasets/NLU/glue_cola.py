from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset


class GLUECoLADataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        context = f"Judge whether the following sentence is grammatically correct or incorrect: {example['sentence']}\nAnswer:"
        target = "Correct" if example["label"] == 1 else "Incorrect"
        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["Correct", "Incorrect"],
        }