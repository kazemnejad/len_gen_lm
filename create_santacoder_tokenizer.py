import os
from pathlib import Path

from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer


def main():
    os.environ["HF_HOME"] = "/raid/hf_home"
    Path(os.environ["HF_HOME"]).mkdir(exist_ok=True, parents=True)

    num_proc = int(os.environ.get("APP_NUM_PROC", 128))
    network_storage_path = Path(os.environ.get("APP_NETWORK_PATH", "/raid/"))

    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder", use_fast=True)

    # Set the bos token
    # This token is not used in our pretraining datasets because we dont use jupyter notebooks.
    # So, it is safe to use it as the bos token.
    tokenizer.bos_token = "<jupyter_start>"

    # Set the template for the tokenizer
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{tokenizer.bos_token} $A {tokenizer.eos_token}",
        special_tokens=[
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id),
        ],
    )

    # Save the subsampled dataset
    tokenizer.save_to_disk(network_storage_path / "santacoder_tokenizer")


if __name__ == "__main__":
    main()
