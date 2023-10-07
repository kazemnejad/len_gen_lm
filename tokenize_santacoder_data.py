import os
from itertools import chain
from pathlib import Path

import transformers
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer
from transformers.testing_utils import CaptureLogger


def main():
    num_proc = int(os.environ.get("APP_NUM_PROC", 200))
    node_storage = Path(os.environ["APP_NODE_PATH"])
    network_storage_path = Path(os.environ["APP_NETWORK_PATH"])

    # Load the tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(
        node_storage / "santacoder_tokenizer", use_fast=True
    )
    dataset = load_from_disk(
        str(node_storage / "santacoder_subsampled"), keep_in_memory=True
    )

    # Split into train and validation
    datasets: DatasetDict = dataset.train_test_split(
        test_size=0.05, shuffle=True, seed=42, keep_in_memory=True
    )
    datasets.flatten_indices(num_proc=num_proc, keep_in_memory=True)

    text_column_name = "content"
    id_column_name = "doc_id"
    lang_name_column_name = "lang"

    bos_token_id = tokenizer.bos_token_id

    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(
                examples[text_column_name], add_special_tokens=True, truncation=False
            )
        return output

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=[text_column_name, lang_name_column_name],
        keep_in_memory=True,
        desc="Running tokenizer on dataset",
    )

    CONTEXT_SIZE = 1024

    block_size = (
        CONTEXT_SIZE - 1
    )  # the bos token will be added to the beginning of each example

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_input_ids = list(chain(*examples["input_ids"]))
        concatenated_doc_ids = list(
            chain(
                *[
                    [doc_id + 1] * len(input_ids)
                    for doc_id, input_ids in zip(
                        examples[id_column_name], examples["input_ids"]
                    )
                ]
            )
        )
        assert len(concatenated_input_ids) == len(concatenated_doc_ids)

        total_length = len(concatenated_input_ids)
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        input_id_chunks = [
            concatenated_input_ids[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        doc_id_chunks = [
            concatenated_doc_ids[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]

        # Add BOS token to the beginning of every example
        input_id_chunks = [[bos_token_id] + input_ids for input_ids in input_id_chunks]
        doc_id_chunks = [[doc_ids[0]] + doc_ids for doc_ids in doc_id_chunks]

        result = {"input_ids": input_id_chunks, "attention_mask": doc_id_chunks}
        result["labels"] = result["input_ids"].copy()

        return result

    datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Grouping texts",
    )

    datasets.save_to_disk(network_storage_path / f"santacoder_data_tokenized_{CONTEXT_SIZE}")


if __name__ == "__main__":
    main()
