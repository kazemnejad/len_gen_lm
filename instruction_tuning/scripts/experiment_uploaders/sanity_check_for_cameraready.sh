#!/bin/bash

python scripts/experiment_uploaders/sanity_check.py \
  --pe "pe_rotary" \
  --dataset_name "s2s_reverse" \
  --ds_split "mc2x_tr20_ts40" \
  --finetune "1b" \
  --tags 'sanityCh_for_cameraready'

python scripts/experiment_uploaders/sanity_check.py \
  --pe "pe_alibi" \
  --dataset_name "s2s_reverse" \
  --ds_split "mc2x_tr20_ts40" \
  --finetune "1b" \
  --tags 'sanityCh_for_cameraready'

python scripts/experiment_uploaders/sanity_check.py \
  --pe "pe_none" \
  --dataset_name "s2s_reverse" \
  --ds_split "mc2x_tr20_ts40" \
  --finetune "1b" \
  --tags 'sanityCh_for_cameraready'