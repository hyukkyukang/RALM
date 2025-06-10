from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset


class GLUEQNLIDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(
        self, example: Dict[str, Any]
    ) -> Dict[str, Any]:
        context = f"Does the given sentence logically entail the question? Answer with 'Yes' for entailment or 'No' if it does not.\nQuestion: {example['question']}\nSentence: {example['sentence']}\nAnswer:"
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
