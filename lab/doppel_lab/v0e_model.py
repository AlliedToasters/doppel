"""Doppel v0e: v0d with z encoded from a set of the player's earlier games (docs/SCIENCE.md §8, "Doppel v0e").

Design (final):
  vectors    per earlier game and side, precomputed once (v0e_vec, v0e_data): Maia-3's mean pooled embedding
             (1024, rating-blind 1500 vs 1500), its policy scalars, the rating sweep as ll(1000) - ll(1500) and
             ll(2000) - ll(1500), and 12 surface stats (think, clock, result, resign / flag / mate, time control).
             No ratings: strength must come from the moves (the players suite scores z rating-blind).
  context    the player's k most recent earlier games (R2: within-player time split, target never in its
             context), k = 0 for a first game or with p 0.05, else log-uniform 1..40 capped by the history.
  encoder    q(z | context set, cond): each game vector -> d=128 (two linear maps, inputs standardised with
             train statistics held as buffers), a cond token (target ratings, time control) prepended, a 2-layer
             pre-norm transformer with no positional encoding (permutation-invariant), the cond token's output
             -> (mu, logvar). k = 0 uses the prior itself, so a new player gets p(z | cond) and KL 0.
  prior      p(z | rating, opponent rating, time control) as v0d; z in R^32 (v0d 8: the evidence is now up to
             40 games, and Maia's averaged embedding needs ~64 PCA dims for its 77% re-ID).
  KL         free bits 0.5 nats per sample (v0d 3.0, which never bound) and beta 0.25: z is the player's, so its
             KL is shared across that player's future games, while each sample charges it to one target game.
  heads      residual, think and resign as v0d (v0_model), conditioned on the 32-d z, initialised from the v0d
             checkpoint: every shape-matching tensor is copied, and the z columns of the first layers that take
             z start at 0, so step 0 is v0d with z ignored.

Gradient rule (v0_model, memory "think-time-head"): all losses are imitation (policy CE, the prior-mixture
marginal, think NLL, resign BCE) plus the KL, so nothing outcome-driven reaches the think head. The context
games' results are encoder inputs (R2 evidence), not losses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from doppel_lab.v0_data import CLK_FEATS, TEMPO_FEATS
from doppel_lab.v0_model import Prior, Residual, ResignHead, ThinkHead, gaussian
from doppel_lab.v0e_data import CTX_FEATS, EMB_DIM

ZE = 32


class SetEncoder(nn.Module):
    def __init__(self, d: int = 128, layers: int = 2, heads: int = 4, cond_dim: int = 4, z: int = ZE):
        super().__init__()
        for k, n in (("emb_mu", EMB_DIM), ("feat_mu", CTX_FEATS)):
            self.register_buffer(k, torch.zeros(n))
        for k, n in (("emb_sd", EMB_DIM), ("feat_sd", CTX_FEATS)):
            self.register_buffer(k, torch.ones(n))
        self.emb_in, self.feat_in = nn.Linear(EMB_DIM, d), nn.Linear(CTX_FEATS, d)
        self.cls = nn.Sequential(nn.Linear(cond_dim, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.out = nn.Linear(d, 2 * z)
        nn.init.normal_(self.out.weight, std=1e-2)   # q starts at ~N(0, I), the zero-initialised prior: KL ~ 0
        nn.init.zeros_(self.out.bias)

    def set_norm(self, path: Path):
        s = np.load(path)
        for k in ("emb_mu", "emb_sd", "feat_mu", "feat_sd"):
            getattr(self, k).copy_(torch.from_numpy(s[k]).float())

    def forward(self, emb, feat, mask, cond):
        e = (emb.float() - self.emb_mu) / self.emb_sd
        f = ((feat.float() - self.feat_mu) / self.feat_sd).clamp(-8, 8)
        x = torch.cat([self.cls(cond)[:, None], self.emb_in(e) + self.feat_in(f)], 1)
        m = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
        h = self.tf(x, src_key_padding_mask=~m)
        return gaussian(self.out(h[:, 0]))


class DoppelV0e(nn.Module):
    def __init__(self, trunk: nn.Module, n_prior_samples: int = 2, z: int = ZE):
        super().__init__()
        self.trunk, self.z_dim = trunk, z
        self.prior, self.encoder, self.residual = Prior(z=z), SetEncoder(z=z), Residual(z=z)
        self.think, self.resign = ThinkHead(trunk.dim, z=z), ResignHead(trunk.dim, z=z)
        self.S = n_prior_samples

    def heads_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("trunk.") and p.requires_grad]

    def posterior(self, b: dict):
        mu_q, lv_q = self.encoder(b["ctx_emb"], b["ctx_feat"], b["ctx_mask"], b["cond"])
        mu_p, lv_p = self.prior(b["cond"])
        has = (b["nctx"] > 0)[:, None]
        return torch.where(has, mu_q, mu_p), torch.where(has, lv_q, lv_p)


def init_from_v0d(model: DoppelV0e, path: Path) -> dict:
    """Copy v0d's residual / think / resign weights. Layers that take z: copy the non-z columns, zero the z ones.
    Prior and encoder start fresh (new z, new evidence). Returns {copied, partial, skipped} key lists."""
    src = torch.load(path, map_location="cpu", weights_only=False)["model"]
    dst = model.state_dict()
    rep = {"copied": [], "partial": [], "skipped": []}
    td = model.trunk.dim
    # key -> (slice of non-z input columns in v0d, same in v0e): z sits at the end (think, resign) or first (ctx)
    zcols = {"residual.ctx.weight": "first", "think.net.0.weight": "last", "resign.net.0.weight": "last"}
    nz = {"think.net.0.weight": td + CLK_FEATS + TEMPO_FEATS, "resign.net.0.weight": td + 3 + CLK_FEATS}
    with torch.no_grad():
        for k, v in src.items():
            if not k.startswith(("residual.", "think.", "resign.")) or k not in dst:
                rep["skipped"].append(k)
                continue
            if dst[k].shape == v.shape:
                dst[k].copy_(v)
                rep["copied"].append(k)
            elif k in zcols and v.shape[0] == dst[k].shape[0] and (
                    k == "residual.ctx.weight" or v.shape[1] == nz[k] + 8):   # v0d: z = 8, same trunk
                dst[k].zero_()
                if zcols[k] == "first":          # residual.ctx: [z, elo_self, elo_opp]
                    dst[k][:, -2:] = v[:, -2:]
                else:
                    dst[k][:, :nz[k]] = v[:, :nz[k]]
                rep["partial"].append(k)
            else:
                rep["skipped"].append(k)
    model.load_state_dict(dst)
    return rep


def contrastive(model: DoppelV0e, b: dict, tau: float = 0.1, gen: torch.Generator | None = None):
    """v0f identity loss: split each sample's context set (k >= 2) into two disjoint random halves, encode each
    with the set encoder, and ask the posterior mean of one half to pick out the same player's other half among
    the batch (symmetric InfoNCE on cosine similarity). Both views get a cond drawn from a random other sample,
    so neither the target's ratings nor its time control can match the views; other samples of the same player
    are dropped from the negatives. Returns (loss, in-batch top-1 accuracy, number of samples used)."""
    m = b["ctx_mask"]
    use = b["nctx"] >= 2
    if use.sum() < 2:
        z = torch.zeros((), device=m.device)
        return z, z, 0
    emb, feat, m, pid = b["ctx_emb"][use], b["ctx_feat"][use], m[use], b["pid"][use]
    n = len(pid)
    r = torch.rand(m.shape, device=m.device, generator=gen).masked_fill(~m, 2.0)
    rank = r.argsort(1).argsort(1)                     # random order of the valid slots, padding last
    k = m.sum(1, keepdim=True)
    a = m & (rank < k // 2)
    bb = m & (rank >= k // 2)
    cond = b["cond"][torch.randint(0, len(b["cond"]), (2, n), device=m.device, generator=gen)]
    mu_a, _ = model.encoder(emb, feat, a, cond[0])
    mu_b, _ = model.encoder(emb, feat, bb, cond[1])
    fa, fb = F.normalize(mu_a.float(), dim=-1), F.normalize(mu_b.float(), dim=-1)
    sim = fa @ fb.T / tau
    same = (pid[:, None] == pid[None, :]) & ~torch.eye(n, dtype=torch.bool, device=m.device)
    sim = sim.masked_fill(same, -1e9)
    y = torch.arange(n, device=m.device)
    loss = 0.5 * (F.cross_entropy(sim, y) + F.cross_entropy(sim.T, y))
    acc = (sim.argmax(1) == y).float().mean()
    return loss, acc.detach(), n
