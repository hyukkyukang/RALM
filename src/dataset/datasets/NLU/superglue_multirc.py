from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset


class SuperGLUEMultiRCDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(
        self, example: Dict[str, Any]
    ) -> Dict[str, Any]:
        # We'll flatten this structure for text-to-text format

        passage_text = example["passage"]["text"]
        questions = example["passage"]["questions"]

        # We'll create a separate example for each question-answer pair
        transformed_examples = []

        for question in questions:
            question_text = question["question"]

            for answer in question["answers"]:
                answer_text = answer["text"]

                # Create the context
                context = f"Passage: {passage_text}\nQuestion: {question_text}\nIs the following answer correct? {answer_text}\nAnswer:"

                # Get the label (0 or 1)
                if "label" in answer:
                    target = "Yes" if answer["label"] == 1 else "No"
                else:
                    # For test set where labels might not be available
                    target = "Unknown"

                transformed_examples.append(
                    {
                        "text": f"{context} {target}",
                        "context": context,
                        "target": target,
                        "choices": ["Yes", "No"],
                        "idx": answer.get("idx", 0),
                        "question_idx": question.get("idx", 0),
                        "passage_idx": example.get("idx", 0),
                    }
                )

        return transformed_examples
