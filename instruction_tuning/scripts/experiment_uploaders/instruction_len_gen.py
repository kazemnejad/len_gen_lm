import os
import re
import uuid
from collections import defaultdict
from typing import Dict, Tuple, List, Any

import fire

POSITIONAL_ENCODINGS = [
    "pe_none",
    "pe_alibi",
    "pe_rotary",
]

# List of (DS, DS_SPLIT) tuples
ALL_DATASETS = [
    ("octa", "instance_len"),
    ("octa", "query_len"),
    ("octa", "response_len"),
]

DS_TO_DS_SPLITS = defaultdict(list)
for ds, ds_split in ALL_DATASETS:
    DS_TO_DS_SPLITS[ds].append(ds_split)

# Sort DS_SPLITS
for ds in DS_TO_DS_SPLITS:
    DS_TO_DS_SPLITS[ds] = sorted(DS_TO_DS_SPLITS[ds])


def generate_boolean_configs(config):
    configs = []
    for i in range(len(config)):
        new_config = config.copy()
        new_config[i] = not config[i]
        configs.append(new_config)
    return configs


def modify_array(arr: List[Any], idx: int, val: Any) -> List[Any]:
    arr[idx] = val
    return arr


def generate_all_scratchpad_configs(ds_name, split_name) -> List[Dict[str, Any]]:
    configs = [{"include_scratchpad": False}]
    return configs


def main(
        dataset_name: str = None,
        ds_split: str = None,
        pe: str = None,
        base_config: str = None,
        sweep_config: str = None,
        launcher: str = None,
        dry_run: bool = False,
        force: bool = False,
        tags: str = None,
        finetune: str = "1b",
):
    if base_config is None:
        base_config = "configs/t5_dec_base.jsonnet"
        if finetune == "1b":
            base_config = "configs/ft_t5_dec_1b_instruct_tune.jsonnet"
            print("Using finetune config (1b)")
        elif finetune == "large":
            base_config = "configs/ft_t5_dec_large.jsonnet"
            print("Using finetune config (large)")

    if sweep_config is None:
        sweep_config = "configs/sweeps/no_sweep.jsonnet"

    if launcher is None:
        launcher = "upload_experiment_with_manual_hp_post_script.sh"

    exp_ids: Dict[str, Tuple[str, str, str]] = {}

    if pe is not None:
        pos_encoding_list = pe.split(",")
    else:
        pos_encoding_list = POSITIONAL_ENCODINGS

    if dataset_name is not None:
        ds_list = dataset_name.split(",")
        ds_list = [(ds, split) for ds in ds_list for split in DS_TO_DS_SPLITS[ds]]
        if ds_split is not None:
            ds_list = [(ds, split) for ds, split in ds_list if split == ds_split]
    else:
        ds_list = ALL_DATASETS

    gpu_class_to_experiment_ids = defaultdict(list)
    experiment_id_to_info = {}

    for ds, ds_split in ds_list:
        print(f"Generating configs for {ds} {ds_split}...")
        for pe in pos_encoding_list:
            for scratchpad_config in generate_all_scratchpad_configs(ds, ds_split):
                if scratchpad_config["include_scratchpad"]:
                    raise NotImplementedError()
                else:
                    scratchpad_config_str = ""
                    scratchpad_config_filename = "no_scratchpad"

                if tags is None:
                    tags = f"instruction_tune,instruction_tune_{ds}"

                cmd = f"scripts/{launcher}"
                cmd += f" --dataset {ds}"
                cmd += f" --split {ds_split}"
                cmd += f' --configs "{base_config},configs/models/{pe}.jsonnet{scratchpad_config_str}"'
                cmd += f" --sweep_configs {sweep_config}"
                cmd += f' --commands "hp_step --eval_split valid"'
                cmd += f' --env "APP_SEED=42"'
                cmd += f" --tags {tags}"
                if not force:
                    cmd += f" --post_script scripts/manual_sweep_launch_best_run_all.sh"
                else:
                    cmd += f" --post_script scripts/manual_sweep_launch_best_run_all_force.sh"

                if not dry_run:
                    output = os.popen(cmd).read()
                else:
                    random_exp_id = str(uuid.uuid4())[0:8]
                    output = f"Exp Key: {random_exp_id}\n"

                # Get the experiment id from the output using a regex
                try:
                    exp_id = re.search(r"Exp Key: (.*)\n", output).group(1)
                    exp_ids[exp_id] = (ds_split, pe, scratchpad_config_filename)
                except Exception as e:
                    print(f"Failed to get exp_id from output: {output}")
                    raise e

                print(f"Experiment id: {exp_id}")

                experiment_id_to_info[exp_id] = {
                    "scratchpad_config": scratchpad_config_filename,
                    "ds": ds,
                    "ds_split": ds_split,
                    "pe": pe,
                }

    # Print experiment ids
    print("Experiment ids:")
    print(",".join(exp_ids.keys()))


if __name__ == "__main__":
    fire.Fire(main)
