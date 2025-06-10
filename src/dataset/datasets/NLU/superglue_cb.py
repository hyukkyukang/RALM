from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset


class SuperGLUECBDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(
        self, example: Dict[str, Any]
    ) -> Dict[str, Any]:
        context = (
            f"Read the following statements carefully:\n\n"
            f"Premise: {example['premise']}\n\n"
            f"Hypothesis: {example['hypothesis']}\n\n"
            f"Does the premise logically entail the hypothesis?\n"
            f"Choose from: 'Yes' (Entailment), 'No' (Contradiction), or 'Maybe' (Neutral).\n"
            f"Answer:"
        )

        # Map label to text
        label_map = {0: "Yes", 1: "No", 2: "Maybe"}
        target = label_map[example["label"]]

        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["Yes", "No", "Maybe"],
            "idx": example.get("idx", 0),
        }
