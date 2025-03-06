from typing import *
from src.dataset.datasets.NLU.text_to_text import TextToTextDataset


class GLUEMNLIDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        context = f"Classify the relationship between the given premise and hypothesis as either Entailment, Neutral, or Contradiction:\nPremise: {example['premise']}\nHypothesis: {example['hypothesis']}\nAnswer:"
        # Set the target label
        if example["label"] == 0:
            target = "Entailment"
        elif example["label"] == 1:
            target = "Neutral"
        else:
            target = "Contradiction"
        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["Entailment", "Neutral", "Contradiction"],
        }