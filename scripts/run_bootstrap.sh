#!/usr/bin/env bash
set -e
DATASET=${1:-mimic3}
A=${2:?path to HH-PMR test_predictions.npz}
B=${3:?path to baseline test_predictions.npz}
for METRIC in jaccard f1 map ddi; do
  python -m hhpmr.bootstrap --a "$A" --b "$B" --metric "$METRIC" --n_boot 1000 --out "runs/$DATASET/bootstrap_$METRIC.json"
done
