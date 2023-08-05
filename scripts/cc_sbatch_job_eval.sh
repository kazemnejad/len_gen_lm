#!/bin/bash

module load StdEnv/2020 gcc/9.3.0 python/3.9 arrow/11
module load httpproxy

source ~/scratch/len_gen_lm/env/bin/activate


export TRANSFORMERS_OFFLINE=1

cd ~/repos/len_gen_lm/
./run_inference.sh "$@"