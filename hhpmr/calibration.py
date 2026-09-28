import torch
import torch.nn.functional as F


def predict_sets(probs, elig, tau):
    eligible = elig > 0
    pred = (probs >= tau) & eligible
    has_candidates = eligible.any(dim=-1)
    empty = (~pred.any(dim=-1)) & has_candidates
    if empty.any():
        masked = probs.masked_fill(~eligible, -1.0)
        top = masked.argmax(dim=-1)
        fallback = F.one_hot(top, probs.shape[-1]).bool()
        pred = torch.where(empty.unsqueeze(-1), fallback, pred)
    return pred


def mean_jaccard(pred, targets):
    target_bool = targets > 0
    inter = (pred & target_bool).sum(dim=-1).float()
    union = (pred | target_bool).sum(dim=-1).float()
    return (inter / union.clamp(min=1.0)).mean().item()


def fit_temperature(logits, targets, elig, max_iter=100):
    logits = logits.detach().float()
    targets = targets.detach().float()
    weights = elig.detach().float()
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_t)
        loss = F.binary_cross_entropy_with_logits(
            logits / temperature, targets, weight=weights, reduction="sum"
        ) / weights.sum().clamp(min=1.0)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).item())


def select_threshold(probs, targets, elig, grid=None):
    if grid is None:
        grid = [round(0.05 * k, 2) for k in range(1, 20)]
    best_tau, best_score = 0.5, -1.0
    for tau in grid:
        score = mean_jaccard(predict_sets(probs, elig, tau), targets)
        if score > best_score:
            best_score, best_tau = score, tau
    return best_tau
