from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset


class GLUESST2Dataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        context = f"Judge whether the following movie review is positive or negative: {example['sentence']}\nAnswer:"
        target = "Positive" if example["label"] == 1 else "Negative"
        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["Positive", "Negative"],
        }