#!/usr/bin/env bash
set -e
DATASET=${1:-mimic3}
DATA_DIR=${2:-data/final_data}
ICD=${3:-9}
COMMON="--dataset $DATASET --data_dir $DATA_DIR --default_icd_version $ICD"
for SEED in 0 1 2 3 4; do
  for LAMBDA in 0 0.001 0.01 1; do
    python -m hhpmr.train $COMMON --run_name "lddi$LAMBDA" --lambda_ddi "$LAMBDA" --seed "$SEED"
  done
  for L in 1 5; do
    python -m hhpmr.train $COMMON --run_name "L$L" --recent_size "$L" --seed "$SEED"
  done
done
for LAMBDA in 0 0.001 0.01 1; do
  python -m hhpmr.aggregate --dir "runs/$DATASET/lddi$LAMBDA"
done
for L in 1 5; do
  python -m hhpmr.aggregate --dir "runs/$DATASET/L$L"
done
