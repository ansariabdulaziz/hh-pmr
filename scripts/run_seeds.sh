#!/usr/bin/env bash
set -e
DATASET=${1:-mimic3}
DATA_DIR=${2:-data/final_data}
ICD=${3:-9}
for SEED in 0 1 2 3 4; do
  python -m hhpmr.train --dataset "$DATASET" --data_dir "$DATA_DIR" --default_icd_version "$ICD" --run_name full --seed "$SEED"
done
python -m hhpmr.aggregate --dir "runs/$DATASET/full"
