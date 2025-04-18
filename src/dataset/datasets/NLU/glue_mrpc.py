from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset


def text_to_text_transform_for_mrpc(example: Dict[str, Any]) -> Dict[str, Any]:
    context = f"Are the following two sentences semantically equivalent?\nSentence1: {example['sentence1']}\nSentence2: {example['sentence2']}\nAnswer:"
    target = "Yes" if example["label"] == 1 else "No"
    return {
        "text": f"{context} {target}",
        "context": context,
        "target": target,
        "choices": ["Yes", "No"],
    }


class GLUEMRPCDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(
        self, example: Dict[str, Any]
    ) -> Dict[str, Any]:
        context = f"Are the following two sentences semantically equivalent?\nSentence1: {example['sentence1']}\nSentence2: {example['sentence2']}\nAnswer:"
        target = "Yes" if example["label"] == 1 else "No"
        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["Yes", "No"],
        }
