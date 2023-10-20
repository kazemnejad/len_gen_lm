#!/bin/bash

set -e

if [ ! -f best_run.json ]; then
  # If best_run.json does not exist, then it means there were no sweeps
  # So, we just run the base config
  echo "best_run.json does not exist. Running base config."
  # Make sure HP_EXP_CONFIG is set
  if [ -z "$HP_EXP_CONFIG" ]; then
    echo "HP_EXP_CONFIG is not set"
    exit 1
  fi
  CONFIGS_STR="${HP_EXP_CONFIG}"
else
  CONFIGS_STR="best_run.json,configs/hp_base.jsonnet,configs/final.jsonnet"

  python scripts/manual_sweep.py \
    --sweep_name $SWEEP_NAME \
    --sweep_root_dir $SWEEP_ROOT_DIR \
    --sweep_configs $SWEEP_CONFIGS \
    fail_if_sweep_not_complete
fi

RUN_ID_PREFIX=$(python scripts/manual_sweep.py \
  --sweep_name $SWEEP_NAME \
  --sweep_root_dir "$SWEEP_ROOT_DIR" \
  --sweep_configs $SWEEP_CONFIGS \
  generate_deterministic_run_id --run_name "best_run")

SEEDS="256788 234054 146317"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

for SEED in $SEEDS; do
  export APP_DIRECTORY=$SWEEP_ROOT_DIR/exps/
  export APP_EXPERIMENT_NAME="best_run_seed_${SEED}"
  export APP_SEED=$SEED
  export WANDB_JOB_TYPE=best_run_seed_exp
  export WANDB_RUN_ID="${RUN_ID_PREFIX}__${SEED}"
    
#  torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
#    src/main.py --configs $CONFIGS_STR \
#    train --eval_split valid
#
#  CUDA_VISIBLE_DEVICES=0 python src/main.py --configs $CONFIGS_STR \
#    predict
  CUDA_VISIBLE_DEVICES=0 python src/main.py --configs $CONFIGS_STR \
    predict --split valid --force

  CUDA_VISIBLE_DEVICES=0 python src/main.py --configs $CONFIGS_STR \
    combine_pred --split valid --force

  CUDA_VISIBLE_DEVICES=0 python src/main.py --configs $CONFIGS_STR \
    analyze_all --split valid

#  CUDA_VISIBLE_DEVICES=0 python src/main.py --configs $CONFIGS_STR \
#    combine_pred
#
#  CUDA_VISIBLE_DEVICES=0 python src/main.py  --configs $CONFIGS_STR \
#    analyze_all
#
  CUDA_VISIBLE_DEVICES=0 python src/main.py --configs $CONFIGS_STR \
    predict --split test --force

  CUDA_VISIBLE_DEVICES=0 python src/main.py --configs $CONFIGS_STR \
    combine_pred --split test --force

  CUDA_VISIBLE_DEVICES=0 python src/main.py --configs $CONFIGS_STR \
    analyze_all --split test --force

done