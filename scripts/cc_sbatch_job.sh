#!/bin/bash

module load StdEnv/2020 gcc/9.3.0 python/3.9 arrow/11
module load httpproxy

source ~/scratch/len_gen_lm/env/bin/activate

cd ~/repos/len_gen_lm/

export TRANSFORMERS_OFFLINE=1

./run_training.sh "$@"