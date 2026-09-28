# HH-PMR

Implementation of HH-PMR: Hierarchy- and History-Aware Personalized Medication Recommendation with Safety-Aware Set Prediction.

## Install

```
pip install -r requirements.txt
```

## Data

`--data_dir` must contain `records_final.pkl`, `voc_final.pkl` and `ddi_A_final.pkl`.

- `records_final.pkl`: list of patients; each patient is a list of visits; each visit is `[diagnosis_indices, procedure_indices, medication_indices]`.
- `voc_final.pkl`: dict with `diag_voc`, `pro_voc`, `med_voc`, each exposing `idx2word`.
- `ddi_A_final.pkl`: binary DDI adjacency over the medication vocabulary.

Diagnosis vocabulary entries may be plain codes, `9:<code>` / `10:<code>` strings, or `(code, version)` tuples. Plain codes use `--default_icd_version`. An optional `--hierarchy_file` JSON of the form `{"9": {"<code>": depth}, "10": {...}}` overrides the built-in ICD depth rules.

## Train and evaluate

```
python -m hhpmr.train --dataset mimic3 --data_dir data/final_data --seed 0
```

Defaults: embedding 64, two self-attention layers with four heads, two directed graph layers, dropout 0.2, AdamW, learning rate 5e-3, weight decay 0.01, batch size 16, at most 200 epochs, early stopping on validation Jaccard with patience 20, L = 3, lambda_ddi = 0.1, patient-disjoint 70/10/20 split.

Outputs per run are written to `runs/<dataset>/<run_name>/seed<seed>/`: `config.json`, `best.pt`, `history.json`, `results.json`, `test_predictions.npz`.

## Experiments

```
bash scripts/run_seeds.sh mimic3 data/final_data 9
bash scripts/run_ablations.sh mimic3 data/final_data 9
bash scripts/run_sensitivity.sh mimic3 data/final_data 9
```

`results.json` contains the main metrics, calibration parameters, the history-sparsity breakdown with mean gate values, the polypharmacy breakdown, and the long-tail recall breakdown. `python -m hhpmr.aggregate --dir runs/mimic3/full` reports mean and standard deviation across seeds.

## Ablation flags

| Variant | Flag |
|---|---|
| -Spec | `--no_spec` |
| -HistGate | `--no_hist_gate` |
| -EHR+ | `--no_ehr_plus` |
| -Reuse | `--no_reuse` |
| -DDI | `--lambda_ddi 0` |
| -Cal | `python -m hhpmr.evaluate --run_dir <run_dir> --no_calibration --tag no_cal` |

## Patient-cluster bootstrap

```
python -m hhpmr.bootstrap --a runs/mimic3/full/seed0/test_predictions.npz --b <baseline>/test_predictions.npz --metric jaccard --n_boot 1000
```

Both files must hold `pred`, `probs`, `targets`, `patient_id` for the same test visits in the same order. Supported metrics: `jaccard`, `f1`, `map`, `ddi`.
