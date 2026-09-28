import argparse
import os

import torch


def get_parser():
    parser = argparse.ArgumentParser("HH-PMR")
    parser.add_argument("--dataset", type=str, default="mimic3")
    parser.add_argument("--data_dir", type=str, default="data/final_data")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--hierarchy_file", type=str, default=None)
    parser.add_argument("--default_icd_version", type=int, default=9, choices=[9, 10])
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--recent_size", type=int, default=3)
    parser.add_argument("--lambda_ddi", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_spec", action="store_true")
    parser.add_argument("--no_hist_gate", action="store_true")
    parser.add_argument("--no_ehr_plus", action="store_true")
    parser.add_argument("--no_reuse", action="store_true")
    parser.add_argument("--no_calibration", action="store_true")
    return parser


def variant_name(args):
    parts = []
    if args.no_spec:
        parts.append("no_spec")
    if args.no_hist_gate:
        parts.append("no_histgate")
    if args.no_ehr_plus:
        parts.append("no_ehrplus")
    if args.no_reuse:
        parts.append("no_reuse")
    if args.no_calibration:
        parts.append("no_cal")
    if args.lambda_ddi != 0.1:
        parts.append("lddi" + format(args.lambda_ddi, "g"))
    if args.recent_size != 3:
        parts.append("L" + str(args.recent_size))
    return "_".join(parts) if parts else "full"


def resolve_run_dir(args):
    name = args.run_name or variant_name(args)
    return os.path.join(args.output_dir, args.dataset, name, "seed" + str(args.seed))
