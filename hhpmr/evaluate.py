import argparse
import json
import os

import torch

from .pipeline import build_model, full_evaluation, setup, write_json


def main():
    parser = argparse.ArgumentParser("HH-PMR evaluation")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--no_calibration", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    cli = parser.parse_args()

    with open(os.path.join(cli.run_dir, "config.json"), "r") as handle:
        config = json.load(handle)
    if cli.device:
        config["device"] = cli.device
    config["no_calibration"] = bool(cli.no_calibration or config.get("no_calibration", False))
    args = argparse.Namespace(**config)

    ctx = setup(args)
    model = build_model(args, ctx)
    state = torch.load(os.path.join(cli.run_dir, "best.pt"), map_location=args.device)
    model.load_state_dict(state)

    suffix = "_" + cli.tag if cli.tag else ""
    results = full_evaluation(args, ctx, model, cli.run_dir, "test_predictions" + suffix + ".npz")
    write_json(os.path.join(cli.run_dir, "results" + suffix + ".json"), results)
    print(results["test"])


if __name__ == "__main__":
    main()
