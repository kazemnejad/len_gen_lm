import os
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer


def main():
    num_proc = int(os.environ.get("APP_NUM_PROC", 8))
    node_storage = Path(os.environ["APP_NODE_PATH"])
    network_storage_path = Path(os.environ["APP_NETWORK_PATH"])

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", use_fast=True
    )

    ds = load_from_disk(str(node_storage / "dolma_subsampled_20B"))

    # Shuffle the dataset
    ds = ds.shuffle(seed=42, writer_batch_size=10000)
    ds = ds.flatten_indices(num_proc=num_proc)

    # Subsample the dataset
    # We only need 20B tokens. Given that the orig dataset is 3084B tokens,
    # we need to take 0.00648508 of the dataset.
    sub_sample_size = int(len(ds) * 0.00648508)
    subsampled = ds.select(range(sub_sample_size))

    # Save the subsampled dataset
    subsampled.save_to_disk(network_storage_path / "dolma_subsampled_20B")


if __name__ == "__main__":
    main()
