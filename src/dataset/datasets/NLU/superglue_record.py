from typing import *

from src.dataset.datasets.NLU.text_to_text import TextToTextDataset


class SuperGLUEReCoRDDataset(TextToTextDataset):
    def _preprocess_fn_for_text_to_text(
        self, example: Dict[str, Any]
    ) -> Dict[str, Any]:
        # ReCoRD has a passage with highlighted entities and a query with a placeholder
        # We need to create examples for each entity candidate

        passage = example["passage"]
        query = example["query"]
        entities = example["entities"]

        # Find the placeholder in the query (usually "@placeholder")
        placeholder = "@placeholder"

        # Create a list to store all transformed examples
        transformed_examples = []

        # For each entity, create an example
        for entity in entities:
            # Replace the placeholder with the entity
            filled_query = query.replace(placeholder, entity)

            # Create the context
            context = f"Passage: {passage}\nQuestion: {filled_query}\nIs this statement correct?"

            # Check if this entity is in the answers
            if "answers" in example and example["answers"]:
                is_correct = entity in example["answers"]
                target = "Yes" if is_correct else "No"
            else:
                # For test set where answers might not be available
                target = "Unknown"

            transformed_examples.append(
                {
                    "text": f"{context} {target}",
                    "context": context,
                    "target": target,
                    "choices": ["Yes", "No"],
                    "idx": example.get("idx", 0),
                    "entity": entity,
                }
            )

        return transformed_examples
