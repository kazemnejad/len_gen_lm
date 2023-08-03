#!/bin/bash

chmod a+x ./env.sh
source ./env.sh


# TRANSFORMERS_CACHE and HF_DATASETS_CACHE
export TRANSFORMERS_CACHE=$APP_EXP_DIR/transformers_cache
export HF_DATASETS_CACHE=$APP_EXP_DIR/datasets_cache

# Read pe_type from command line
PE_TYPE=$1
# Make sure PE_TYPE is passed in
if [ -z "$PE_TYPE" ]
then
    echo "PE_TYPE is empty. Please pass in PE_TYPE as the first argument."
    exit 1
fi
# Read model_size from command line
MODEL_SIZE=$2
# If model_size is not passed in, use default
if [ -z "$MODEL_SIZE" ]
then
    MODEL_SIZE="100m"
fi

# Set Logger values
export CUSTOM_EXPERIMENT_KEY="pppprefixt5deconlywikitext103${PE_TYPE}${MODEL_SIZE}"
export CUSTOM_EXPERIMENT_NAME="WikiText103 $MODEL_SIZE $PE_TYPE"
export COMET_PROJECT_NAME=len-gen-lm

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Running training script with $NUM_GPUS GPUs"
echo "PE_TYPE: $PE_TYPE"
echo "MODEL_SIZE: $MODEL_SIZE"

# Use torchrun to run the training script on multiple GPUs
torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py \
    configs/main.json \
    $PE_TYPE \
    $MODEL_SIZE