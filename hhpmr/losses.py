import torch.nn.functional as F


def set_loss(logits, targets, elig, valid):
    per_item = F.softplus(logits) - targets * logits
    denom = elig.sum(dim=-1).clamp(min=1.0)
    per_visit = (per_item * elig).sum(dim=-1) / denom
    mask = valid.float()
    return (per_visit * mask).sum() / mask.sum().clamp(min=1.0)


def ddi_loss(probs, elig, ddi_adj, valid):
    masked = elig * probs
    per_visit = 0.5 * (masked * (masked @ ddi_adj)).sum(dim=-1)
    mask = valid.float()
    return (per_visit * mask).sum() / mask.sum().clamp(min=1.0)


def total_loss(logits, probs, targets, elig, valid, ddi_adj, lambda_ddi):
    supervised = set_loss(logits, targets, elig, valid)
    if lambda_ddi == 0:
        return supervised
    return supervised + lambda_ddi * ddi_loss(probs, elig, ddi_adj, valid)
