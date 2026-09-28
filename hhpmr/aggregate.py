import argparse
import glob
import json
import os

import numpy as np


def collect_numbers(obj, prefix, store):
    if isinstance(obj, dict):
        for key, value in obj.items():
            collect_numbers(value, prefix + "." + key if prefix else key, store)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        store.setdefault(prefix, []).append(float(obj))


def main():
    parser = argparse.ArgumentParser("HH-PMR seed aggregation")
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--file", type=str, default="results.json")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--ddof", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "seed*", args.file)))
    if not paths:
        raise FileNotFoundError(os.path.join(args.dir, "seed*", args.file))

    store = {}
    for path in paths:
        with open(path, "r") as handle:
            payload = json.load(handle)
        payload.pop("counts", None)
        collect_numbers(payload, "", store)

    summary = {"n_seeds": len(paths)}
    for key, values in sorted(store.items()):
        array = np.asarray(values)
        summary[key] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=args.ddof)) if len(array) > args.ddof else 0.0,
            "n": int(len(array)),
        }

    target = args.out or os.path.join(args.dir, "summary_" + os.path.splitext(args.file)[0] + ".json")
    with open(target, "w") as handle:
        json.dump(summary, handle, indent=2)
    for key in ("test.jaccard", "test.f1", "test.prauc", "test.map", "test.ddi", "test.ece", "test.brier", "test.avg_med"):
        if key in summary:
            print("%-14s %.4f +- %.4f" % (key, summary[key]["mean"], summary[key]["std"]))


if __name__ == "__main__":
    main()
