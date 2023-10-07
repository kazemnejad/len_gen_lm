#!/bin/bash

# Variables
SOURCE_DIR=/raid/len_gen_lm_exps
DEST_DIR=/scratch/

# Infinite loop to keep running the sync command every 5 minutes
while true; do
    rsync -avzh $SOURCE_DIR $DEST_DIR
    sleep 300  # Sleep for 5 minutes
done