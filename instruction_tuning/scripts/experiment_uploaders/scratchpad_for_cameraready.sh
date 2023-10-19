#!/bin/bash


# Addition task with full scratchpad format
#python scripts/experiment_uploaders/scratchpad.py \
#  --pe "pe_alibi" \
#  --dataset_name "s2s_addition" \
#  --tgt_scratchpad_cfg "no_scratchpad" \
#  --finetune \
#  --tags 'scratch_for_rebuttal'

python scripts/experiment_uploaders/scratchpad.py \
  --pe "pe_rotary" \
  --dataset_name "s2s_addition" \
  --tgt_scratchpad_cfg "no_scratchpad" \
  --finetune "1b" \
  --tags 'scratch_for_cameraready'

python scripts/experiment_uploaders/scratchpad.py \
  --pe "pe_alibi" \
  --dataset_name "s2s_addition" \
  --tgt_scratchpad_cfg "no_scratchpad" \
  --finetune "1b" \
  --tags 'scratch_for_cameraready'

python scripts/experiment_uploaders/scratchpad.py \
  --pe "pe_none" \
  --dataset_name "s2s_addition" \
  --tgt_scratchpad_cfg "no_scratchpad" \
  --finetune "1b" \
  --tags 'scratch_for_cameraready'

#python scripts/experiment_uploaders/scratchpad.py \
#  --pe "pe_alibi" \
#  --dataset_name "s2s_addition" \
#  --tgt_scratchpad_cfg "i0_c1_o1_v0_r0" \
#  --finetune \
#  --tags 'scratch_for_rebuttal'

#python scripts/experiment_uploaders/scratchpad.py \
#  --pe "pe_none" \
#  --dataset_name "s2s_addition" \
#  --tgt_scratchpad_cfg "i0_c1_o1_v0_r0" \
#  --finetune \
#  --tags 'scratch_for_rebuttal'