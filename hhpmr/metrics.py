import numpy as np
from sklearn.metrics import average_precision_score

HISTORY_ORDER = ["0", "1-2", "3-5", ">5"]
POLYPHARMACY_ORDER = ["low", "medium", "high"]
FREQUENCY_ORDER = ["head", "torso", "tail"]


def jaccard_f1(pred, targets):
    inter = (pred * targets).sum(axis=-1)
    union = ((pred + targets) > 0).sum(axis=-1)
    jaccard = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    denom = pred.sum(axis=-1) + targets.sum(axis=-1)
    f1 = np.where(denom > 0, 2 * inter / np.maximum(denom, 1), 0.0)
    return jaccard, f1


def per_visit_ap(probs, targets):
    scores = np.zeros(len(targets))
    for n in range(len(targets)):
        if targets[n].sum() > 0:
            scores[n] = average_precision_score(targets[n], probs[n])
    return scores


def mean_average_precision(probs, targets):
    return float(per_visit_ap(probs, targets).mean())


def prauc_macro(probs, targets):
    scores = []
    for m in range(targets.shape[1]):
        column = targets[:, m]
        if column.sum() > 0:
            scores.append(average_precision_score(column, probs[:, m]))
    return float(np.mean(scores)) if scores else 0.0


def ddi_components(pred, ddi):
    numerator = 0.5 * ((pred @ ddi) * pred).sum(axis=-1)
    size = pred.sum(axis=-1)
    denominator = size * (size - 1) / 2.0
    return numerator, denominator


def ddi_rate(pred, ddi, eps=1e-6):
    numerator, denominator = ddi_components(pred, ddi)
    return float(numerator.sum() / (denominator.sum() + eps))


def brier_score(probs, targets):
    return float(np.mean((probs - targets) ** 2))


def expected_calibration_error(probs, targets, n_bins=15):
    p = probs.reshape(-1)
    y = targets.reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(p)
    error = 0.0
    for k in range(n_bins):
        if k == n_bins - 1:
            in_bin = (p >= edges[k]) & (p <= edges[k + 1])
        else:
            in_bin = (p >= edges[k]) & (p < edges[k + 1])
        count = in_bin.sum()
        if count == 0:
            continue
        error += (count / total) * abs(p[in_bin].mean() - y[in_bin].mean())
    return float(error)


def summarize(pred, probs, targets, ddi, n_bins=15):
    jaccard, f1 = jaccard_f1(pred, targets)
    return {
        "n_visits": int(len(pred)),
        "jaccard": float(jaccard.mean()),
        "f1": float(f1.mean()),
        "prauc": prauc_macro(probs, targets),
        "map": mean_average_precision(probs, targets),
        "ddi": ddi_rate(pred, ddi),
        "ece": expected_calibration_error(probs, targets, n_bins),
        "brier": brier_score(probs, targets),
        "avg_med": float(pred.sum(axis=-1).mean()),
    }


def history_labels(prior):
    labels = np.empty(len(prior), dtype=object)
    labels[prior == 0] = "0"
    labels[(prior >= 1) & (prior <= 2)] = "1-2"
    labels[(prior >= 3) & (prior <= 5)] = "3-5"
    labels[prior > 5] = ">5"
    return labels


def history_breakdown(pred, probs, targets, ddi, prior, rho, n_bins=15):
    labels = history_labels(prior)
    result = {}
    for name in HISTORY_ORDER:
        mask = labels == name
        if not mask.any():
            result[name] = None
            continue
        entry = summarize(pred[mask], probs[mask], targets[mask], ddi, n_bins)
        entry["mean_rho"] = None if name == "0" else float(rho[mask].mean())
        result[name] = entry
    return result


def polypharmacy_breakdown(pred, probs, targets, ddi, n_bins=15):
    counts = targets.sum(axis=-1)
    q1, q2 = np.quantile(counts, [1 / 3, 2 / 3])
    labels = np.empty(len(counts), dtype=object)
    labels[counts <= q1] = "low"
    labels[(counts > q1) & (counts <= q2)] = "medium"
    labels[counts > q2] = "high"
    result = {"thresholds": {"q1": float(q1), "q2": float(q2)}}
    for name in POLYPHARMACY_ORDER:
        mask = labels == name
        result[name] = summarize(pred[mask], probs[mask], targets[mask], ddi, n_bins) if mask.any() else None
    return result


def frequency_groups(train_counts):
    order = np.argsort(-np.asarray(train_counts), kind="stable")
    head, torso, tail = np.array_split(order, 3)
    return {"head": head, "torso": torso, "tail": tail}


def long_tail_breakdown(pred, targets, train_counts):
    positives = targets.sum(axis=0)
    hits = (pred * targets).sum(axis=0)
    recall = np.where(positives > 0, hits / np.maximum(positives, 1), np.nan)
    result = {}
    for name, members in frequency_groups(train_counts).items():
        values = recall[members]
        result[name] = {
            "n_medications": int(len(members)),
            "recall": float(np.nanmean(values)) if not np.all(np.isnan(values)) else None,
        }
    return result
