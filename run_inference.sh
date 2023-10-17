#!/bin/bash

# APP_EXP_DIR should be a shared network storage. We save checkpoints and logs here.

export APP_SHARED_STORAGE_PATH=/raid/
export APP_CONFIG_PATH=configs/code_llm.json

# TRANSFORMERS_CACHE and HF_DATASETS_CACHE
export HF_HOME=/raid/hf_home
export TRANSFORMERS_CACHE=/raid/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/raid/.cache/huggingface/datasets

mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE

# Sync checkpoints from network storage to local storage
rsync -avzh /scratch/len_gen_lm_exps /raid/

# Read pe_type from command line
PE_TYPE=$1
# Make sure PE_TYPE is passed in
if [ -z "$PE_TYPE" ]
then
    echo "PE_TYPE is empty. Please pass in PE_TYPE as the first argument."
    exit 1
fi

export APP_EXP_DIR=/scratch_$PE_TYPE/len_gen_lm_exps

# Set Logger values
export WANDB_PROJECT="santacoder"
export WANDB_NAME="SantaCoder 1B $PE_TYPE"
export WANDB_RUN_ID="final-santacoder-1b-${PE_TYPE}-eval"
export WANDB_RESUME="allow"

export OMP_NUM_THREADS=100

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Run sync_checkpoints_to_network.sh in the background
chmod +x sync_checkpoints_to_network.sh
./sync_checkpoints_to_network.sh &

echo "Running inference script with $NUM_GPUS GPUs"
echo "PE_TYPE: $PE_TYPE"

torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    inference_llm.py \
    configs/code_llm.json \
    $PE_TYPE >> "$APP_EXP_DIR/$PE_TYPE.log" 2>&1