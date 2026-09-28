import argparse
import json

import numpy as np

from . import metrics as M
from .data import load_ddi

EPS = 1e-6


def visit_terms(metric, data, ddi):
    pred = data["pred"].astype(np.float32)
    targets = data["targets"].astype(np.float32)
    if metric == "jaccard":
        values = M.jaccard_f1(pred, targets)[0]
        return values, np.ones(len(values))
    if metric == "f1":
        values = M.jaccard_f1(pred, targets)[1]
        return values, np.ones(len(values))
    if metric == "map":
        values = M.per_visit_ap(data["probs"], targets)
        return values, np.ones(len(values))
    if metric == "ddi":
        return M.ddi_components(pred, ddi)
    raise ValueError(metric)


def patient_sums(values, inverse, n_patients):
    return np.bincount(inverse, weights=values, minlength=n_patients)


def bootstrap_statistics(terms_a, terms_b, patient_id, n_boot, seed, alpha):
    _, inverse = np.unique(patient_id, return_inverse=True)
    n_patients = int(inverse.max()) + 1
    num_a = patient_sums(terms_a[0], inverse, n_patients)
    den_a = patient_sums(terms_a[1], inverse, n_patients)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_patients, size=(n_boot, n_patients))

    def interval(values):
        return [float(np.percentile(values, 100 * alpha / 2)), float(np.percentile(values, 100 * (1 - alpha / 2)))]

    stat_a = num_a[idx].sum(axis=1) / (den_a[idx].sum(axis=1) + EPS)
    result = {
        "a": {"estimate": float(num_a.sum() / (den_a.sum() + EPS)), "ci": interval(stat_a)},
        "n_patients": n_patients,
        "n_boot": n_boot,
    }
    if terms_b is None:
        return result

    num_b = patient_sums(terms_b[0], inverse, n_patients)
    den_b = patient_sums(terms_b[1], inverse, n_patients)
    stat_b = num_b[idx].sum(axis=1) / (den_b[idx].sum(axis=1) + EPS)
    diff = stat_a - stat_b
    lower = float((diff <= 0).mean())
    upper = float((diff >= 0).mean())
    p_value = min(1.0, max(2 * min(lower, upper), 1.0 / n_boot))
    result["b"] = {"estimate": float(num_b.sum() / (den_b.sum() + EPS)), "ci": interval(stat_b)}
    result["difference"] = {
        "estimate": result["a"]["estimate"] - result["b"]["estimate"],
        "ci": interval(diff),
        "p_value": p_value,
        "p_value_floor": 1.0 / n_boot,
    }
    return result


def main():
    parser = argparse.ArgumentParser("HH-PMR patient-cluster bootstrap")
    parser.add_argument("--a", type=str, required=True)
    parser.add_argument("--b", type=str, default=None)
    parser.add_argument("--metric", type=str, default="jaccard", choices=["jaccard", "f1", "map", "ddi"])
    parser.add_argument("--data_dir", type=str, default="data/final_data")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    data_a = np.load(args.a)
    ddi = load_ddi(args.data_dir) if args.metric == "ddi" else None
    terms_a = visit_terms(args.metric, data_a, ddi)
    terms_b = None
    if args.b:
        data_b = np.load(args.b)
        if not np.array_equal(data_a["patient_id"], data_b["patient_id"]) or not np.array_equal(
            data_a["targets"], data_b["targets"]
        ):
            raise ValueError("prediction files are not aligned on patients and targets")
        terms_b = visit_terms(args.metric, data_b, ddi)

    result = bootstrap_statistics(terms_a, terms_b, data_a["patient_id"], args.n_boot, args.seed, args.alpha)
    result["metric"] = args.metric
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
