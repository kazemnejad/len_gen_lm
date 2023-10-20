import random
from typing import Dict, Any

from data.data_instance_processor.data_instance_processor import DataInstanceProcessor


@DataInstanceProcessor.register("octa")
class OctaDataInstanceProcessor(DataInstanceProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = random.Random(42)

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        targets = self._convert_targets(example[self.target_seq_key])
        source = self._create_input(example)
        example.update(
            {
                self.source_seq_key: source,
                self.target_seq_key: targets,
                "answer": targets.replace("<|endoftext|>", ""),
                "category": self._get_category(example),
            }
        )
        return example

    def _convert_targets(self, targets: str) -> str:
        return targets

    def _create_input(self, example: Dict[str, Any]) -> str:
        # Randomly put the instruction in the beginning or the end with 50% chance
        message_first = self.rng.choice([True, False])
        if message_first:
            return f"{example['message']}\n\n{example['old_contents']}<commit_after>"
        else:
            return f"{example['old_contents']}\n\n{example['message']}<commit_after>"

    def _get_category(self, example: Dict[str, Any]) -> str:
        potential_category_keys = ["category", "cat", "categories", "cats"]
        for key in potential_category_keys:
            if key in example:
                return str(example[key])
        # Compute the category based on the length of the response

        if self.split_name == "response_len":
            length_key = "response_len"
        elif self.split_name == "query_len":
            length_key = "query_len"
        else:
            length_key = "instance_len"

        return str(example[length_key] // 20 * 20)

    def extract_answer_from_prediction(self, prediction: str) -> Any:
        if "<commit_after>" in prediction:
            prediction_str = prediction.split("<commit_after>")[1].replace("<|endoftext|>", "")
        else:
            prediction_str = prediction
        return prediction_str
