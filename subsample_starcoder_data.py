import os
from pathlib import Path

from datasets import load_dataset, DownloadConfig, concatenate_datasets


def subsample(ds, num_samples, num_proc=8):
    """
    Randomly subsample `num_samples` from the dataset `ds`.

    Parameters:
        ds (list or other iterable): The dataset to subsample from.
        num_samples (int): The number of samples to take.

    Returns:
        subsampled_data: The subsampled data
    """
    if num_samples >= len(ds):
        return ds  # or raise a value error depending on use case
    else:
        ds = ds.shuffle(seed=42, writer_batch_size=10000)
        ds = ds.flatten_indices(num_proc=num_proc)
        ds = ds.select(range(num_samples))

        return ds


def main():
    os.environ["HF_HOME"] = "/raid/hf_home"
    Path(os.environ["HF_HOME"]).mkdir(exist_ok=True, parents=True)

    num_proc = int(os.environ.get("APP_NUM_PROC", 128))
    network_storage_path = Path(os.environ.get("APP_NETWORK_PATH", "/raid/"))

    ds_dict = {
        ln: load_dataset(
            "bigcode/starcoderdata",
            data_dir=ln,
            split="train",
            num_proc=num_proc,
            download_config=DownloadConfig(
                resume_download=True, max_retries=30, num_proc=num_proc
            ),
        )
        for ln in [
            "java",
            "github-issues-filtered-structured",
            "python",
            "javascript",
            "git-commits-cleaned",
        ]
    }

    # Remove all columns except for the content and id columns
    for ln, ds in ds_dict.items():
        ds_dict[ln] = ds.remove_columns(
            [
                col_name
                for col_name in ds.column_names
                if col_name not in ["content", "idx"]
            ]
        )

    ds_ratios = {
        "python": 0.4,
        "java": 0.25,
        "javascript": 0.25,
        "github-issues-filtered-structured": 0.05,
        "git-commits-cleaned": 0.05,
    }

    num_rows_dict = {lang_name: len(ds) for lang_name, ds in ds_dict.items()}
    print("num orig. rows:", num_rows_dict)

    # Find the maximum number of samples that can be obtained while maintaining the ratio
    limiting_ds_name = min(
        ds_ratios.keys(),
        key=lambda ds_name: num_rows_dict[ds_name] / ds_ratios[ds_name],
    )
    max_samples = int(num_rows_dict[limiting_ds_name] / ds_ratios[limiting_ds_name])

    # calculate the number of samples per dataset based on the ratios and max_samples
    num_samples_dict = {
        lang_name: int(ratio * max_samples) for lang_name, ratio in ds_ratios.items()
    }

    # Log the number of samples per dataset and original number of samples
    print("num sampled rows:", num_samples_dict)
    print("Total num samples:", sum(num_samples_dict.values()))

    # Compute the percentage of subsampling
    subsample_percentages = {
        lang_name: num_samples_dict[lang_name] / sum(num_samples_dict.values())
        for lang_name in num_samples_dict.keys()
    }
    print("subsample percentages:", subsample_percentages)

    print("Subsampling...")
    # Subsample each dataset
    for lang_name, ds in ds_dict.items():
        print(f"Subsampling {lang_name}...")
        ds_dict[lang_name] = subsample(
            ds, num_samples_dict[lang_name], num_proc=num_proc
        )

    # Add a column to each dataset with the language name
    for lang_name, ds in ds_dict.items():
        ds_dict[lang_name] = ds.map(
            lambda example: {"lang": [lang_name] * len(example["content"])},
            batched=True,
            num_proc=num_proc,
            desc=f"Adding language column to {lang_name}",
        )

    # Concatenate all datasets
    ds = concatenate_datasets(list(ds_dict.values()))

    # Shuffle the dataset
    ds = ds.shuffle(seed=42, writer_batch_size=10000)
    ds = ds.flatten_indices(num_proc=num_proc)

    # Remove the idx column
    ds = ds.remove_columns("id")

    # Add the idx column back in
    ds = ds.map(
        lambda example, i: {"doc_id": i},
        with_indices=True,
        num_proc=num_proc,
        desc="Adding doc id",
    )

    print("Final dataset size:", len(ds))

    # Save the subsampled dataset
    ds.save_to_disk(network_storage_path / "santacoder_subsampled")


if __name__ == "__main__":
    main()
