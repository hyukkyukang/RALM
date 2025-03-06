from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset

class GLUESTSBDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("STSB is not implemented yet.")
        context = f"Determine the similarity between the following two sentences: {example['sentence1']} and {example['sentence2']}\nAnswer:"
        target = "Positive" if example["label"] == 1 else "Negative"
        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["Positive", "Negative"],
        }
