import os
from pathlib import Path

from datasets import load_dataset, DownloadConfig
from transformers import GPT2TokenizerFast


def main():
    os.environ["HF_HOME"] = "/raid/hf_home"
    Path(os.environ["HF_HOME"]).mkdir(exist_ok=True, parents=True)

    num_proc = int(os.environ.get("APP_NUM_PROC", 24))
    network_storage_path = Path(os.environ.get("APP_NETWORK_PATH", "/raid/"))

    ds = load_dataset(
        "bigcode/starcoderdata",
        num_proc=num_proc,
        download_config=DownloadConfig(resume_download=True, max_retries=30, num_proc=num_proc),
        revision="9fc30b578cedaec69e47302df72cf00feed7c8c4",
    )

    ds.save_to_disk(network_storage_path / "dolma_orig")

    print(len(ds))
    print(ds)
    print(ds.features)
    print(ds["train"][0])

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.add_special_tokens()


if __name__ == "__main__":
    main()
