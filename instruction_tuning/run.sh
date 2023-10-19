#!/bin/bash


export APP_SEED=42
export APP_DS_SPLIT=$FT_DATASET_SPLIT

export SWEEP_NAME="SW-ft_t5_dec_1b_instruct_tune_pe_${FT_PE_TYPE}_octa___data-octa-instance_len"
export SWEEP_CONFIGS='configs/sweeps/no_sweep.jsonnet'
export CAPTURE_LOG=1
export SWEEP_ROOT_DIR=experiments/$SWEEP_NAME
export HP_EXP_CONFIG="configs/ft_t5_dec_1b_instruct_tune.jsonnet,configs/models/pe_${FT_PE_TYPE}.jsonnet,configs/data/octa.jsonnet"
mkdir -p $SWEEP_ROOT_DIR

export WANDB_RUN_GROUP="SW-ft_t5_dec_1b_instruct_tune_pe_${FT_PE_TYPE}_octa___data-octa-instance_len"
export WANDB_TAGS=sweep,manual_sweep,launched_by_ngc,instruction_tune,instruction_tune_octa

chmod a+x scripts/manual_sweep_agent.sh
./scripts/manual_sweep_agent.sh


chmod a+x scripts/manual_sweep_launch_best_run_all.sh
./scripts/manual_sweep_launch_best_run_all.sh


echo "Experiment finished!"
