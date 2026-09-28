#!/usr/bin/env bash
set -e
DATASET=${1:-mimic3}
DATA_DIR=${2:-data/final_data}
ICD=${3:-9}
COMMON="--dataset $DATASET --data_dir $DATA_DIR --default_icd_version $ICD"
for SEED in 0 1 2 3 4; do
  python -m hhpmr.train $COMMON --run_name no_spec --no_spec --seed "$SEED"
  python -m hhpmr.train $COMMON --run_name no_histgate --no_hist_gate --seed "$SEED"
  python -m hhpmr.train $COMMON --run_name no_ehrplus --no_ehr_plus --seed "$SEED"
  python -m hhpmr.train $COMMON --run_name no_reuse --no_reuse --seed "$SEED"
  python -m hhpmr.train $COMMON --run_name no_ddi --lambda_ddi 0 --seed "$SEED"
  python -m hhpmr.evaluate --run_dir "runs/$DATASET/full/seed$SEED" --no_calibration --tag no_cal
done
for NAME in no_spec no_histgate no_ehrplus no_reuse no_ddi; do
  python -m hhpmr.aggregate --dir "runs/$DATASET/$NAME"
done
python -m hhpmr.aggregate --dir "runs/$DATASET/full" --file results_no_cal.json
