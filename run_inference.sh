#!/bin/bash

# APP_EXP_DIR should be a shared network storage. We save checkpoints and logs here.

export APP_SHARED_STORAGE_PATH=/raid/
export APP_CONFIG_PATH=configs/code_llm_eval.json

# TRANSFORMERS_CACHE and HF_DATASETS_CACHE
export HF_HOME=/raid/hf_home
export TRANSFORMERS_CACHE=/raid/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/raid/.cache/huggingface/datasets

mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE

# Read pe_type from command line
PE_TYPE=$1
# Make sure PE_TYPE is passed in
if [ -z "$PE_TYPE" ]
then
    echo "PE_TYPE is empty. Please pass in PE_TYPE as the first argument."
    exit 1
fi

export APP_EXP_DIR="/scratch_${PE_TYPE}/len_gen_lm_exps"
export APP_SHARED_STORAGE_PATH="/scratch_${PE_TYPE}/"

# Set Logger values
export WANDB_PROJECT="santacoder"
export WANDB_NAME="SantaCoder 1B $PE_TYPE (PPL Eval)"
export WANDB_RUN_ID="final-santacoder-1b-${PE_TYPE}-eval"
export WANDB_RESUME="allow"

export OMP_NUM_THREADS=100

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Running inference script with $NUM_GPUS GPUs"
echo "PE_TYPE: $PE_TYPE"

torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    inference_llm.py \
    configs/code_llm_eval.json \
    $PE_TYPE >> "$APP_SHARED_STORAGE_PATH/$PE_TYPE.log.eval" 2>&1