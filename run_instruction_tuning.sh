#!/bin/bash

# Read pe_type from command line
PE_TYPE=$1
# Make sure PE_TYPE is passed in
if [ -z "$PE_TYPE" ]
then
    echo "PE_TYPE is empty. Please pass in PE_TYPE as the first argument."
    exit 1
fi

DATASET_SPLIT=$2
# Make sure DATASET_SPLIT is passed in
if [ -z "$DATASET_SPLIT" ]
then
    echo "DATASET_SPLIT is empty. Please pass in DATASET_SPLIT as the second argument."
    exit 1
fi

export PRETRAINED_MODEL_PATH="/scratch/len_gen_lm_exps/1b__${PE_TYPE}/final-model"
export NETWORK_SHARED_STORAGE_PATH="/scratch/"

INSTRUCT_TUNE_CODEBASE=instruction_tuning
LOG_FILE="${NETWORK_SHARED_STORAGE_PATH}/instruction_tuning_exps/${DATASET_SPLIT}__${PE_TYPE}.log"

mkdir -p $NETWORK_SHARED_STORAGE_PATH/instruction_tuning_exps/

ln -snf $NETWORK_SHARED_STORAGE_PATH/instruction_tuning_exps/ $INSTRUCT_TUNE_CODEBASE/experiments
ln -snf $PRETRAINED_MODEL_PATH $INSTRUCT_TUNE_CODEBASE/experiments/t5_dec_only_1b_santacoder_$PE_TYPE

# Install requirements
apt-get install -y python3.10-venv

# Create virtual environment
python -m venv /raid/instruction_tuning_venv
source /raid/instruction_tuning_venv/bin/activate

# Install requirements
pip3 install torch torchvision torchaudio >> $LOG_FILE 2>&1
pip3 install tokenizers sacrebleu >> $LOG_FILE 2>&1

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip3 install -r $INSTRUCT_TUNE_CODEBASE/requirements.txt >> $LOG_FILE 2>&1
pip3 install --upgrade wandb

# Download dataset
wandb artifact get --root $INSTRUCT_TUNE_CODEBASE/data kzmnjd/len_gen/data-octa-$DATASET_SPLIT >> $LOG_FILE 2>&1

export FT_PE_TYPE=$PE_TYPE
export FT_DATASET_SPLIT=$DATASET_SPLIT

# Run instruction tuning pipeline
cd $INSTRUCT_TUNE_CODEBASE

chmod a+x run.sh
./run.sh >> $LOG_FILE 2>&1