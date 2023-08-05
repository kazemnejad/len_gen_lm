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
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import json
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any
from pathlib import Path

import datasets
import evaluate
import torch
from datasets import load_dataset
from tqdm import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    T5TokenizerFast,
    T5Config,
    PreTrainedTokenizer,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
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

from torch.utils.data import Dataset as TorchDataset, DataLoader


class PerplexityEvaluationDataset(TorchDataset):
    def __init__(
        self,
        tokenized_dataset: datasets.Dataset,
        block_size: int,
        stride: int,
        bos_token_id: int,
    ):
        self.block_size = block_size
        self.stride = stride
        self.bos_token_id = bos_token_id

        assert "input_ids" in tokenized_dataset.column_names
        assert "doc_id" in tokenized_dataset.column_names

        # Concat all input_ids from all examples together
        all_input_ids = []
        all_doc_ids = []
        for example in tokenized_dataset:
            all_input_ids.extend(example["input_ids"])
            all_doc_ids.extend([example["doc_id"]] * len(example["input_ids"]))

        self.all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        self.all_doc_ids = torch.tensor(all_doc_ids, dtype=torch.long)
        assert self.all_input_ids.shape == self.all_doc_ids.shape

        self.dataset_len = len(range(0, self.all_input_ids.shape[-1] - block_size + 1, stride))

        del all_input_ids
        del all_doc_ids

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, i):
        begin_loc = i * self.stride
        end_loc = begin_loc + self.block_size
        input_ids = self.all_input_ids[begin_loc:end_loc]
        doc_ids = self.all_doc_ids[begin_loc:end_loc]

        assert input_ids.shape == doc_ids.shape
        assert input_ids.shape[-1] == self.block_size

        # Prepend the bos_token_id
        input_ids = torch.cat([torch.tensor([self.bos_token_id]), input_ids], dim=0)
        doc_ids = torch.cat(
            [torch.tensor([doc_ids[0]]), doc_ids], dim=0
        )  # doc id is repeated

        assert input_ids.shape == doc_ids.shape

        labels = input_ids.clone()

        # Mask out all tokens but the last one.
        labels[:-1] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": doc_ids,
        }


def data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if "attention_mask" in features[0]:
        doc_ids = [f.pop("attention_mask") for f in features]
    else:
        doc_ids = None

    batch = default_data_collator(features)
    if doc_ids is not None:
        seq_length = batch["input_ids"].shape[-1]
        causal_mask = torch.arange(seq_length)[:, None] >= torch.arange(seq_length)
        doc_mask = [
            ((di[:, None] == di) & causal_mask).int()
            for di in doc_ids
        ]
        attention_mask = torch.stack(doc_mask, dim=0)
        batch["attention_mask"] = attention_mask
    return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
    if len(sys.argv) >= 3:
        model_args.pe_type = sys.argv[2]
        logger.info("Using pe_type: %s", model_args.pe_type)
    training_args.output_dir = os.path.join(
        os.environ.get("APP_EXP_DIR", "experiments"), model_args.pe_type
    )

    # Compute the batch_size_per_device based on world_size
    # Target batch size is the optimization batch size.
    # So, we will divide target batch size by the number of processes
    world_size = training_args.world_size

    assert world_size == 1, "Only single GPU inference is supported"

    # Check if we have pe_type passed in as an argument
    if len(sys.argv) >= 3:
        model_args.pe_type = sys.argv[2]
        logger.info("Using pe_type: %s", model_args.pe_type)

    if len(sys.argv) >= 4:
        # Read the model size from the command line
        model_size = sys.argv[3]
        model_size_to_t5_config = {
            "100m": "t5-base",
            "300m": "t5-large",
            "1b": "t5-3b",
        }
        model_args.config_name = model_size_to_t5_config[model_size.lower()]
        model_args.tokenizer_name = model_args.config_name
        logger.info(f"Using model size: {model_size}({model_args.config_name})")

    training_args.output_dir = os.path.join(
        os.environ.get("APP_EXP_DIR", "experiments"),
        f"{model_args.pe_type}__{model_args.config_name}",
    )

    logger.info(f"Using output_dir: {training_args.output_dir}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    final_model_dir = os.path.join(training_args.output_dir, "final-model")

    # tokenizer = AutoTokenizer.from_pretrained("t5-large")

    tokenizer = AutoTokenizer.from_pretrained(final_model_dir, use_fast=True)
    config = AutoConfig.from_pretrained(final_model_dir)

    training_args: TrainingArguments
    if training_args.local_rank == 0:
        logger.info(config)
        logger.info(tokenizer)

    test_dataset = raw_datasets["test"]
    column_names = test_dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    BOS_TOKEN_ID = tokenizer.additional_special_tokens_ids[-1]
    logger.info(f"Using BOS token id: {BOS_TOKEN_ID}")

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(
                examples[text_column_name], add_special_tokens=True, truncation=False
            )
            output["input_ids"] = [
                [BOS_TOKEN_ID] + o for o in output["input_ids"]  # add bos token
            ]
            del output["attention_mask"]  # no need for attention mask
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_dataset = test_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            # add document id to each example
            tokenized_dataset = tokenized_dataset.map(
                lambda example, i: {"doc_id": i},
                with_indices=True,
                desc="Adding doc id",
            )
        else:
            raise NotImplementedError("Streaming not implemented")

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Load the model
    model = modeling.CustomDecoderOnlyT5.from_pretrained(
        final_model_dir, position_encoding_type=model_args.pe_type
    )
    assert model.config.position_encoding_type == model_args.pe_type

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        callbacks=[],
    )

    block_sizes = [128, 256, 384, 512, 640, 768, 896, 1024, 1152]
    results_dir = Path(training_args.output_dir) / "perplexity_results"
    if trainer.is_world_process_zero():
        results_dir.mkdir(parents=True, exist_ok=True)

    for blk_sz in tqdm(block_sizes):
        result_file = results_dir / f"ppl_{blk_sz}.json"
        try:
            with result_file.open() as f:
                result = json.load(f)
                if len(result) > 0:
                    if trainer.is_world_process_zero():
                        logger.info("Skipping block size: %s", blk_sz)
                    continue
        except Exception:
            pass

        if trainer.is_world_process_zero():
            logger.info("Evaluating block size: %s", blk_sz)

        block_size = blk_sz - 1
        eval_dataset = PerplexityEvaluationDataset(
            tokenized_dataset,
            block_size=block_size,
            stride=1,
            bos_token_id=BOS_TOKEN_ID,
        )

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        metrics["eval_samples"] = len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)

        if trainer.is_world_process_zero():
            with result_file.open("w") as f:
                json.dump(result, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    # block_sizes = range(16, 3, 16)
    main()
