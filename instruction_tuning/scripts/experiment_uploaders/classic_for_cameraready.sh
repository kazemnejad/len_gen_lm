#!/bin/bash

python scripts/experiment_uploaders/classic_len_gen.py \
  --dataset_name "scan" \
  --pe "pe_rotary" \
  --ds_split "mdlen_tr25_ts48" \
  --finetune "1b" \
  --tags 'classic_for_cameraready'

python scripts/experiment_uploaders/classic_len_gen.py \
  --dataset_name "scan" \
  --pe "pe_alibi" \
  --ds_split "mdlen_tr25_ts48" \
  --finetune "1b" \
  --tags 'classic_for_cameraready'

python scripts/experiment_uploaders/classic_len_gen.py \
  --dataset_name "scan" \
  --pe "pe_none" \
  --ds_split "mdlen_tr25_ts48" \
  --finetune "1b" \
  --tags 'classic_for_cameraready'