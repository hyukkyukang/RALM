from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset

class GLUERTEDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        context = f"Does Sentence1 logically entail Sentence2? Answer with 'Yes' for entailment or 'No' if it does not.\nSentence1: {example['sentence1']}\nSentence2: {example['sentence2']}\nAnswer:"
        # Set the target label
        if example["label"] == 0:
            target = "Yes"
        elif example["label"] == 1:
            target = "No"
        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["Yes", "No"],
        }