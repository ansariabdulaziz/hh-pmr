import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics as M
from .calibration import fit_temperature, mean_jaccard, predict_sets, select_threshold
from .data import load_raw, split_patients
from .dataset import PatientDataset, make_collate
from .graph import build_graph_arrays, medication_visit_counts
from .hierarchy import build_specificity_vector, load_depth_maps
from .losses import total_loss
from .model import HHPMR


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(clean(payload), handle, indent=2)


def setup(args):
    seed_everything(args.seed)
    records, diag_words, proc_words, med_words, ddi = load_raw(args.data_dir)
    train_records, val_records, test_records = split_patients(records, args.split_seed)
    num_diag, num_proc, num_meds = len(diag_words), len(proc_words), len(med_words)

    depth_maps = load_depth_maps(args.hierarchy_file) if args.hierarchy_file else None
    spec_vec, coverage = build_specificity_vector(diag_words, args.default_icd_version, depth_maps)
    graph = build_graph_arrays(train_records, num_meds)
    train_med_counts = medication_visit_counts(train_records, num_meds)
    max_gap = max((len(p) - 1 for p in train_records), default=1)

    collate = make_collate(num_meds, num_diag, num_proc)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            PatientDataset(train_records),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            generator=generator,
        ),
        "val": DataLoader(
            PatientDataset(val_records), batch_size=args.batch_size, shuffle=False, collate_fn=collate
        ),
        "test": DataLoader(
            PatientDataset(test_records), batch_size=args.batch_size, shuffle=False, collate_fn=collate
        ),
    }
    counts = {
        "patients": {"train": len(train_records), "val": len(val_records), "test": len(test_records)},
        "visits": {
            "train": sum(len(p) for p in train_records),
            "val": sum(len(p) for p in val_records),
            "test": sum(len(p) for p in test_records),
        },
        "vocabulary": {"diagnosis": num_diag, "procedure": num_proc, "medication": num_meds},
        "hierarchy_coverage": coverage,
    }
    return SimpleNamespace(
        num_diag=num_diag,
        num_proc=num_proc,
        num_meds=num_meds,
        spec_vec=spec_vec,
        graph=graph,
        ddi=ddi,
        train_med_counts=train_med_counts,
        max_gap=max_gap,
        loaders=loaders,
        counts=counts,
    )


def build_model(args, ctx):
    model = HHPMR(
        num_diag=ctx.num_diag,
        num_proc=ctx.num_proc,
        num_meds=ctx.num_meds,
        dim=args.emb_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        recent_size=args.recent_size,
        delta_t_max=float(max(ctx.max_gap, 1)),
        use_spec=not args.no_spec,
        use_hist_gate=not args.no_hist_gate,
        use_ehr_plus=not args.no_ehr_plus,
        use_reuse=not args.no_reuse,
    )
    model.set_specificity(torch.tensor(ctx.spec_vec))
    model.set_graph(
        torch.tensor(ctx.graph["a_out"]),
        torch.tensor(ctx.graph["a_in"]),
        torch.tensor(ctx.graph["b_out"]),
        torch.tensor(ctx.graph["b_in"]),
    )
    return model.to(args.device)


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def valid_visits(batch):
    return batch["visit_mask"] & (batch["med_multihot"].sum(dim=-1) > 0)


def forward_batch(model, batch):
    return model(
        batch["diag_ids"],
        batch["diag_mask"],
        batch["proc_ids"],
        batch["proc_mask"],
        batch["med_multihot"],
    )


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    keys = ["logits", "targets", "elig", "prior", "rho", "patient_id"]
    store = {key: [] for key in keys}
    for batch in loader:
        batch = to_device(batch, device)
        out = forward_batch(model, batch)
        valid = valid_visits(batch)
        store["logits"].append(out["logits"][valid].cpu())
        store["targets"].append(batch["med_multihot"][valid].cpu())
        store["elig"].append(out["elig"][valid].cpu())
        store["prior"].append(batch["visit_index"][valid].cpu())
        store["rho"].append(out["rho"][valid].cpu())
        store["patient_id"].append(batch["patient_id"][valid].cpu())
    return {key: torch.cat(values) for key, values in store.items()}


def train_model(args, ctx, model, run_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ddi = torch.tensor(ctx.ddi, device=args.device)
    checkpoint = os.path.join(run_dir, "best.pt")
    best_score, best_epoch, waiting = -1.0, -1, 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running, batches = 0.0, 0
        for batch in ctx.loaders["train"]:
            batch = to_device(batch, args.device)
            out = forward_batch(model, batch)
            loss = total_loss(
                out["logits"],
                out["probs"],
                batch["med_multihot"],
                out["elig"],
                valid_visits(batch),
                ddi,
                args.lambda_ddi,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            batches += 1

        val = collect(model, ctx.loaders["val"], args.device)
        pred = predict_sets(torch.sigmoid(val["logits"]), val["elig"], 0.5)
        score = mean_jaccard(pred, val["targets"])
        history.append({"epoch": epoch, "train_loss": running / max(batches, 1), "val_jaccard": score})
        print("epoch %d loss %.4f val_jaccard %.4f" % (epoch, running / max(batches, 1), score), flush=True)

        if score > best_score:
            best_score, best_epoch, waiting = score, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            waiting += 1
            if waiting >= args.patience:
                break

    write_json(os.path.join(run_dir, "history.json"), history)
    return history, best_epoch


def calibrate(args, val):
    if args.no_calibration:
        temperature = 1.0
    else:
        temperature = fit_temperature(val["logits"], val["targets"], val["elig"])
    probs = val["elig"] * torch.sigmoid(val["logits"] / temperature)
    threshold = select_threshold(probs, val["targets"], val["elig"])
    return temperature, threshold


def decode(out, temperature, threshold):
    probs = out["elig"] * torch.sigmoid(out["logits"] / temperature)
    pred = predict_sets(probs, out["elig"], threshold)
    return (
        pred.numpy().astype(np.float32),
        probs.numpy().astype(np.float32),
        out["targets"].numpy().astype(np.float32),
    )


def full_evaluation(args, ctx, model, run_dir, prediction_name="test_predictions.npz"):
    val = collect(model, ctx.loaders["val"], args.device)
    test = collect(model, ctx.loaders["test"], args.device)
    temperature, threshold = calibrate(args, val)
    model.set_calibration(temperature, threshold)

    val_pred, val_probs, val_targets = decode(val, temperature, threshold)
    test_pred, test_probs, test_targets = decode(test, temperature, threshold)
    prior = test["prior"].numpy()
    rho = test["rho"].numpy()

    np.savez_compressed(
        os.path.join(run_dir, prediction_name),
        pred=test_pred.astype(np.uint8),
        probs=test_probs,
        targets=test_targets.astype(np.uint8),
        patient_id=test["patient_id"].numpy(),
        prior=prior,
    )

    return {
        "temperature": temperature,
        "threshold": threshold,
        "counts": ctx.counts,
        "val": M.summarize(val_pred, val_probs, val_targets, ctx.ddi, args.n_bins),
        "test": M.summarize(test_pred, test_probs, test_targets, ctx.ddi, args.n_bins),
        "history": M.history_breakdown(test_pred, test_probs, test_targets, ctx.ddi, prior, rho, args.n_bins),
        "polypharmacy": M.polypharmacy_breakdown(test_pred, test_probs, test_targets, ctx.ddi, args.n_bins),
        "long_tail": M.long_tail_breakdown(test_pred, test_targets, ctx.train_med_counts),
    }
