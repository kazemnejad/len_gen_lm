#!/bin/bash

cd ~/scratch

sbatch \
  --gres=gpu:2 \
  -t 20:00:00 \
  --account rrg-bengioy-ad \
  --mem 32G \
  -c 8 \
  --output ~/scratch/len_gen_lm/logs/300m-none.out \
  --error ~/scratch/len_gen_lm/logs/300m-none.err \
  --wrap="~/repos/len_gen_lm/scripts/cc_sbatch_job.sh none 300m"