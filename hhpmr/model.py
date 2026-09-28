import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_CLIP = 1e-6


def masked_softmax(logits, mask):
    masked = logits.masked_fill(~mask, -1e9)
    return torch.softmax(masked, dim=-1) * mask.float()


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, mask):
        logits = self.score(x).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        return (weights * x).sum(dim=-2)


def make_encoder(dim, nhead, num_layers, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=nhead,
        dim_feedforward=dim * 2,
        dropout=dropout,
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)


class CurrentStateEncoder(nn.Module):
    def __init__(self, num_diag, num_proc, dim, nhead, num_layers, dropout, use_spec, attr_vocab_size=1):
        super().__init__()
        self.dim = dim
        self.use_spec = use_spec
        self.diag_emb = nn.Embedding(num_diag + 1, dim, padding_idx=num_diag)
        self.proc_emb = nn.Embedding(num_proc + 1, dim, padding_idx=num_proc)
        self.diag_missing = nn.Parameter(torch.randn(dim) * 0.02)
        self.proc_missing = nn.Parameter(torch.randn(dim) * 0.02)
        self.w_q = nn.Parameter(torch.randn(dim) * 0.02)
        self.register_buffer("spec_vec", torch.zeros(num_diag + 1))
        self.diag_encoder = make_encoder(dim, nhead, num_layers, dropout)
        self.proc_encoder = make_encoder(dim, nhead, num_layers, dropout)
        self.diag_pool = AttentionPool(dim)
        self.proc_pool = AttentionPool(dim)
        self.attr_emb = nn.Embedding(attr_vocab_size, dim)
        self.state_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
        )

    def set_specificity(self, values):
        with torch.no_grad():
            self.spec_vec.copy_(values)

    @staticmethod
    def with_missing_token(emb, mask, missing):
        empty = ~mask.any(dim=-1)
        first = torch.where(
            empty.unsqueeze(-1),
            missing.view(1, 1, -1).expand_as(emb[:, :, 0, :]),
            emb[:, :, 0, :],
        )
        emb = torch.cat([first.unsqueeze(2), emb[:, :, 1:, :]], dim=2)
        mask = torch.cat([(mask[:, :, 0] | empty).unsqueeze(2), mask[:, :, 1:]], dim=2)
        return emb, mask

    @staticmethod
    def encode(encoder, x, mask):
        b, t, l, d = x.shape
        out = encoder(x.reshape(b * t, l, d), src_key_padding_mask=~mask.reshape(b * t, l))
        return out.reshape(b, t, l, d)

    def forward(self, diag_ids, diag_mask, proc_ids, proc_mask, attr_ids=None):
        b, t = diag_ids.shape[:2]
        diag_emb, diag_mask = self.with_missing_token(self.diag_emb(diag_ids), diag_mask, self.diag_missing)
        proc_emb, proc_mask = self.with_missing_token(self.proc_emb(proc_ids), proc_mask, self.proc_missing)
        diag_enc = self.encode(self.diag_encoder, diag_emb, diag_mask)
        proc_enc = self.encode(self.proc_encoder, proc_emb, proc_mask)
        if self.use_spec:
            diag_enc = diag_enc + self.spec_vec[diag_ids].unsqueeze(-1) * self.w_q
        pooled_diag = self.diag_pool(diag_enc, diag_mask)
        pooled_proc = self.proc_pool(proc_enc, proc_mask)
        if attr_ids is None:
            attr_ids = torch.zeros(b, t, dtype=torch.long, device=diag_ids.device)
        attributes = self.attr_emb(attr_ids)
        return self.state_mlp(torch.cat([pooled_diag, pooled_proc, attributes], dim=-1))


class HistoryModule(nn.Module):
    def __init__(self, dim, recent_size, delta_t_max, use_hist_gate):
        super().__init__()
        self.recent_size = recent_size
        self.use_hist_gate = use_hist_gate
        self.register_buffer("delta_t_max", torch.tensor(float(max(delta_t_max, 1.0))))
        self.theta_gamma = nn.Parameter(torch.zeros(()))
        self.gate = nn.Linear(dim * 3, 1)

    def forward(self, h):
        b, t, _ = h.shape
        device = h.device
        t_idx = torch.arange(t, device=device).view(t, 1)
        j_idx = torch.arange(t, device=device).view(1, t)
        causal = (j_idx < t_idx).unsqueeze(0).expand(b, t, t)

        if not self.use_hist_gate:
            count = causal.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
            weights = causal.float() / count
            return torch.bmm(weights, h), weights, torch.ones(b, t, device=device)

        norms = h.norm(dim=-1)
        cosine = torch.bmm(h, h.transpose(1, 2)) / (norms.unsqueeze(2) * norms.unsqueeze(1) + 1e-8)
        gap = (t_idx - j_idx).clamp(min=0).float()
        delta = (torch.log1p(gap) / torch.log1p(self.delta_t_max)).clamp(max=1.0)
        gamma = torch.sigmoid(self.theta_gamma)
        scores = cosine - gamma * delta.unsqueeze(0)

        recent_band = (j_idx >= (t_idx - self.recent_size)).unsqueeze(0).expand(b, t, t)
        recent_mask = causal & recent_band
        distant_mask = causal & ~recent_band
        recent_present = recent_mask.any(dim=-1)
        distant_present = distant_mask.any(dim=-1)

        alpha_recent = masked_softmax(scores, recent_mask)
        alpha_distant = masked_softmax(scores, distant_mask)
        h_recent = torch.bmm(alpha_recent, h)
        h_distant = torch.bmm(alpha_distant, h)

        rho_learned = torch.sigmoid(self.gate(torch.cat([h, h_recent, h_distant], dim=-1))).squeeze(-1)
        rho = torch.where(recent_present & distant_present, rho_learned, recent_present.float())

        h_hist = rho.unsqueeze(-1) * h_recent + (1 - rho).unsqueeze(-1) * h_distant
        weights = rho.unsqueeze(-1) * alpha_recent + (1 - rho).unsqueeze(-1) * alpha_distant
        return h_hist, weights, rho


class DirectedEHRGraph(nn.Module):
    def __init__(self, num_meds, dim, dropout, use_ehr_plus):
        super().__init__()
        self.use_ehr_plus = use_ehr_plus
        self.med_emb = nn.Embedding(num_meds, dim)
        self.self_1 = nn.Linear(dim, dim, bias=False)
        self.out_1 = nn.Linear(dim, dim, bias=False)
        self.in_1 = nn.Linear(dim, dim, bias=False)
        self.self_2 = nn.Linear(dim, dim, bias=False)
        self.out_2 = nn.Linear(dim, dim, bias=False)
        self.in_2 = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(dim * 2, dim)
        for name in ("a_out", "a_in", "b_out", "b_in"):
            self.register_buffer(name, torch.zeros(num_meds, num_meds))

    def set_graph(self, a_out, a_in, b_out, b_in):
        with torch.no_grad():
            self.a_out.copy_(a_out)
            self.a_in.copy_(a_in)
            self.b_out.copy_(b_out)
            self.b_in.copy_(b_in)

    def forward(self):
        if self.use_ehr_plus:
            a_out, a_in = self.a_out, self.a_in
        else:
            a_out, a_in = self.b_out, self.b_in
        h0 = self.med_emb.weight
        h1 = self.dropout(F.relu(self.self_1(h0) + a_out @ self.out_1(h0) + a_in @ self.in_1(h0)))
        h2 = self.dropout(F.relu(self.self_2(h1) + a_out @ self.out_2(h1) + a_in @ self.in_2(h1)))
        return self.project(torch.cat([h0, h2], dim=-1))


class PairwiseScorer(nn.Module):
    def __init__(self, state_dim, med_dim, hidden_dim):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.med_proj = nn.Linear(med_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state, z):
        hidden = F.gelu(self.state_proj(state).unsqueeze(2) + self.med_proj(z).view(1, 1, z.shape[0], -1))
        return self.out(hidden).squeeze(-1)


class LinearGate(nn.Module):
    def __init__(self, state_dim, med_dim):
        super().__init__()
        self.state = nn.Linear(state_dim, 1)
        self.med = nn.Linear(med_dim, 1, bias=False)

    def forward(self, state, z):
        return self.state(state) + self.med(z).view(1, 1, -1)


class HHPMR(nn.Module):
    def __init__(
        self,
        num_diag,
        num_proc,
        num_meds,
        dim=64,
        nhead=4,
        num_layers=2,
        dropout=0.2,
        recent_size=3,
        delta_t_max=1.0,
        use_spec=True,
        use_hist_gate=True,
        use_ehr_plus=True,
        use_reuse=True,
    ):
        super().__init__()
        self.num_meds = num_meds
        self.use_reuse = use_reuse
        self.encoder = CurrentStateEncoder(num_diag, num_proc, dim, nhead, num_layers, dropout, use_spec)
        self.history = HistoryModule(dim, recent_size, delta_t_max, use_hist_gate)
        self.graph = DirectedEHRGraph(num_meds, dim, dropout, use_ehr_plus)
        self.new_scorer = PairwiseScorer(dim * 2, dim, dim)
        self.gate_scorer = LinearGate(dim * 2, dim)
        self.register_buffer("temperature", torch.tensor(1.0))
        self.register_buffer("threshold", torch.tensor(0.5))

    def set_specificity(self, values):
        self.encoder.set_specificity(values)

    def set_graph(self, a_out, a_in, b_out, b_in):
        self.graph.set_graph(a_out, a_in, b_out, b_in)

    def set_calibration(self, temperature, threshold):
        with torch.no_grad():
            self.temperature.fill_(float(temperature))
            self.threshold.fill_(float(threshold))

    def forward(self, diag_ids, diag_mask, proc_ids, proc_mask, med_multihot, attr_ids=None, eligibility=None):
        b, t = diag_ids.shape[:2]
        device = diag_ids.device
        h = self.encoder(diag_ids, diag_mask, proc_ids, proc_mask, attr_ids)
        h_hist, weights, rho = self.history(h)
        z = self.graph()
        state = torch.cat([h, h_hist], dim=-1)
        logit_new = self.new_scorer(state, z)

        if self.use_reuse:
            reuse = torch.bmm(weights, med_multihot).clamp(EPS_CLIP, 1 - EPS_CLIP)
            logit_reuse = torch.log(reuse) - torch.log1p(-reuse)
            chi = (torch.arange(t, device=device) >= 1).float().view(1, t, 1)
            gate = chi * torch.sigmoid(self.gate_scorer(state, z))
            logits = (1 - gate) * logit_new + gate * logit_reuse
        else:
            logits = logit_new

        if eligibility is None:
            elig = torch.ones(b, t, self.num_meds, device=device)
        else:
            elig = eligibility.float()

        return {
            "logits": logits,
            "probs": torch.sigmoid(logits),
            "elig": elig,
            "calibrated_probs": elig * torch.sigmoid(logits / self.temperature),
            "rho": rho,
        }
