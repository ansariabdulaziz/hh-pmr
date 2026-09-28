import os

import torch

from .config import get_parser, resolve_run_dir
from .pipeline import build_model, full_evaluation, setup, train_model, write_json


def main():
    args = get_parser().parse_args()
    run_dir = resolve_run_dir(args)
    os.makedirs(run_dir, exist_ok=True)
    write_json(os.path.join(run_dir, "config.json"), vars(args))

    ctx = setup(args)
    model = build_model(args, ctx)
    _, best_epoch = train_model(args, ctx, model, run_dir)

    state = torch.load(os.path.join(run_dir, "best.pt"), map_location=args.device)
    model.load_state_dict(state)

    results = full_evaluation(args, ctx, model, run_dir)
    results["best_epoch"] = best_epoch
    write_json(os.path.join(run_dir, "results.json"), results)
    print(results["test"])


if __name__ == "__main__":
    main()
