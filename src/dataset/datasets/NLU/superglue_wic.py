from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset


class SuperGLUEWiCDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # Extract key information
        word = example["word"]
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        
        context = (
            f"Determine whether the word '{word}' has the same meaning in both sentences.\n\n"
            f"Sentence 1: {sentence1}\n"
            f"Sentence 2: {sentence2}\n\n"
            f"Does '{word}' mean the same thing in both sentences? Answer 'Yes' or 'No'.\n"
            f"Answer:"
        )
        
        # Map label to text (1 = same meaning, 0 = different meaning)
        target = "Yes" if example["label"] == 1 else "No"
        
        return {
            "text": f"{context} {target}",
            "context": context,
            "target": target,
            "choices": ["Yes", "No"],
            "idx": example.get("idx", 0),
        }