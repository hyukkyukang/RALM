from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset


class SuperGLUERTEDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        context = (
            f"Read the following statements carefully and determine if the first statement supports the second.\n\n"
            f"Premise: {example['premise']}\n\n"
            f"Hypothesis: {example['hypothesis']}\n\n"
            f"Does the premise logically entail the hypothesis? Answer 'Yes' for entailment and 'No' for non-entailment.\n"
            f"Answer:"
        )
        
        # Map label to text (0 = entailment, 1 = not_entailment)
        target = "Yes" if example["label"] == 0 else "No"
        
        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["Yes", "No"],
            "idx": example.get("idx", 0),
        }