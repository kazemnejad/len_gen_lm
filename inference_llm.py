#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
import json

# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import datasets
import numpy as np
import torch
import transformers
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.30.0.dev0")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    pe_type: Optional[str] = field(
        default=None,
        metadata={"help": ("Positional encoding type. ")},
    )

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`"
            )

        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."


import model as modeling


def create_prediction_dataset(
    orig_dataset: Dataset,
    block_size: int,
    bos_token_id: int,
    num_proc: int,
    max_inference_toks: int = 200,
):
    block_size_minus_1 = block_size - 1

    def map_fn(examples: Dict[str, List[Any]]):
        num_examples = len(examples["input_ids"])

        output_dict = {
            "input_ids": [],
            "attention_mask": [],
            "input_ids_len": [],
            "length_bucket": [],
            "doc_id": [],
            "prediction_point": [],
            "labels": [],
        }
        for i in range(num_examples):
            input_ids = examples["input_ids"][i]
            doc_id = examples["doc_id"][i]

            input_ids_len = len(input_ids)

            if (
                examples["length_bucket"][i] < block_size_minus_1
                and examples["length_bucket"][i] < 4500
            ):
                continue

            # for j in range(input_ids_len - block_size_minus_1 + 1):
            for j in range(-1, -max_inference_toks - 1, -1):
                start_idx = j - block_size_minus_1
                end_idx = j + 1
                if start_idx < -input_ids_len:
                    break

                if end_idx == 0:
                    sample_input_ids = input_ids[start_idx:]
                else:
                    sample_input_ids = input_ids[start_idx:end_idx]

                sample_input_ids = np.array([bos_token_id] + sample_input_ids)
                sample_labels = np.array(
                    [-100] * (len(sample_input_ids) - 1) + [sample_input_ids[-1]]
                )
                output_dict["input_ids"].append(sample_input_ids)
                output_dict["labels"].append(sample_labels)
                output_dict["attention_mask"].append(np.ones(len(sample_input_ids)))
                output_dict["input_ids_len"].append(len(sample_input_ids))
                output_dict["length_bucket"].append(examples["length_bucket"][i])
                output_dict["doc_id"].append(doc_id)
                output_dict["prediction_point"].append(-1 - j)

        return output_dict

    ds = orig_dataset.map(
        map_fn,
        batched=True,
        remove_columns=["content", "lang"],
        num_proc=num_proc,
        desc="Creating prediction dataset",
        batch_size=128,
        writer_batch_size=128,
    )

    return ds


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    config_path = os.environ["APP_CONFIG_PATH"]
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=os.path.abspath(config_path)
    )

    training_args.do_train = True
    training_args.do_eval = True

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Check if we have pe_type passed in as an argument
    model_args.pe_type = sys.argv[2]
    logger.info("Using pe_type: %s", model_args.pe_type)

    assert model_args.pe_type in ["none", "alibi", "rotary"]

    model_size = "1b"
    logger.info(f"Using model size: {model_size}")

    training_args.output_dir = os.path.join(
        os.environ.get("APP_EXP_DIR", "experiments"),
        f"1b__{model_args.pe_type}",
    )

    shared_storage_path = os.environ["APP_SHARED_STORAGE_PATH"]

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    final_model_dir = os.path.join(training_args.output_dir, "final-model")

    tokenizer = AutoTokenizer.from_pretrained(final_model_dir, use_fast=True)
    model = modeling.CustomDecoderOnlyT5.from_pretrained(
        final_model_dir, model_args.pe_type, True
    )
    assert model.config.position_encoding_type == model_args.pe_type
    assert model.output_non_reduced_loss == True

    with training_args.main_process_first(desc="dataset map tokenization"):
        test_dataset = load_from_disk(
            os.path.join(
                shared_storage_path,
                "santacoder_subsampled_32M_docs_test_uniform_less_4500",
            ),
        )
        test_dataset = test_dataset.shuffle(seed=42)
        test_dataset = test_dataset.flatten_indices(num_proc=12)
        test_dataset = test_dataset.select(range(3000))

    logger.info(f"Loaded dataset {test_dataset}")
    logger.info(f"Loaded tokenizer {tokenizer}")

    n_params = sum(
        {
            p.data_ptr(): p.numel() for p in model.parameters() if p.requires_grad
        }.values()
    )
    logger.info(
        f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=None,
    )

    # Increase batch size by 128
    #  32,   32, 32,   16,  16, 16
    block_sizes = [
        512,
        640,
        750,
        878,
        1024,
        1200,
        1400,
        1600,
        1800,
        2048,
        2304,
        2560,
        2560,
    ]

    def get_eval_device_batch_size(block_size):
        return {
            512: 64,
            640: 52,
            750: 42,
            878: 32,
            1024: 26,
            1200: 20,
            1400: 20,
            1600: 20,
            1800: 20,
            2048: 16,
            2304: 16,
            2560: 16,
        }[block_size]

    results_dir = Path(shared_storage_path) / "perplexity_results_on_test"
    if trainer.is_world_process_zero():
        results_dir.mkdir(parents=True, exist_ok=True)

    for blk_sz in tqdm(block_sizes):
        result_file = results_dir / f"pred_dataset_{blk_sz}"
        try:
            ds = load_from_disk(result_file)
            if "ppl" in ds.column_names and "losses" in ds.column_names:
                if trainer.is_world_process_zero():
                    logger.info("Skipping block size: %s", blk_sz)
                continue
        except Exception:
            pass

        torch.cuda.empty_cache()

        if trainer.is_world_process_zero():
            logger.info("Evaluating block size: %s", blk_sz)

        with training_args.main_process_first(desc="dataset map tokenization"):
            pred_dataset = create_prediction_dataset(
                test_dataset,
                block_size=blk_sz,
                bos_token_id=tokenizer.bos_token_id,
                num_proc=64,
            )

        if trainer.is_world_process_zero():
            logger.info("Length of pred_dataset: %s", len(pred_dataset))

        batch_size = get_eval_device_batch_size(blk_sz)
        trainer.args.per_device_eval_batch_size = batch_size

        if trainer.is_world_process_zero():
            logger.info("batch_size: %s", batch_size)

        output = trainer.predict(
            pred_dataset,
            ignore_keys=["logits", "past_key_values", "hidden_states", "attentions"],
        )
        trainer.log({"finished": blk_sz})

        if trainer.is_world_process_zero():
            losses = output.predictions
            ppl = np.exp(losses)
            pred_dataset = pred_dataset.remove_columns(
                ["input_ids", "labels", "attention_mask"]
            )
            pred_dataset = pred_dataset.add_column("ppl", np.squeeze(ppl))
            pred_dataset = pred_dataset.add_column("losses", np.squeeze(losses))
            pred_dataset.save_to_disk(result_file)


if __name__ == "__main__":
    main()
