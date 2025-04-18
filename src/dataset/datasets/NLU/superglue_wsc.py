from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset


class SuperGLUEWSCDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(
        self, example: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Extract key information
        text = example["text"]
        pronoun = example["span1_text"]  # The pronoun
        noun = example["span2_text"]  # The noun

        context = (
            f"Read the following sentence and determine if the pronoun refers to the given noun.\n\n"
            f"Sentence: {text}\n\n"
            f"Pronoun: '{pronoun}'\n"
            f"Possible referent: '{noun}'\n\n"
            f"Does '{pronoun}' refer to '{noun}' in this sentence? Answer 'Yes' or 'No'.\n"
            f"Answer:"
        )

        # Map label to text (1 = yes, 0 = no)
        target = "Yes" if example["label"] == 1 else "No"

        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["Yes", "No"],
            "idx": example.get("idx", 0),
        }
