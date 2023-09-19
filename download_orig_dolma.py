import os

from datasets import load_dataset, DownloadConfig


def main():
    num_proc = int(os.environ.get("APP_NUM_PROC", 8))
    ds = load_dataset(
        "allenai/dolma",
        num_proc=num_proc,
        download_config=DownloadConfig(resume_download=True, max_retries=20, num_proc=num_proc),
    )


if __name__ == "__main__":
    main()
