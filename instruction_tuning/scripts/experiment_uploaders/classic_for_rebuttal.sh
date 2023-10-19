#!/bin/bash

#python scripts/experiment_uploaders/classic_len_gen.py \
#  --dataset_name "scan" \
#  --pe "pe_alibi" \
#  --ds_split "mdlen_tr25_ts48" \
#  --finetune \
#  --tags 'classic_for_rebuttal'

python scripts/experiment_uploaders/classic_len_gen.py \
  --dataset_name "scan" \
  --pe "pe_none" \
  --ds_split "mdlen_tr25_ts48" \
  --finetune \
  --tags 'classic_for_rebuttal'