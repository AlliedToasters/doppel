"""Doppel v0 model (brief, "Phase 2 spec: Doppel v0"; docs/SCIENCE.md §5-6).

  trunk      Maia-3 79M, frozen. Its logits are the teacher; its pooled features (8-ply history) and win/draw/loss
             feed the think-time and resign heads. It is a fixed function, so no loss trains it.
  prior      p(z | rating, opponent rating, time control), diagonal Gaussian, z in R^8
  encoder    q(z | one half of a game): a small transformer over oracle-free per-ply features (teacher log-prob
             and rank of the played move, teacher entropy, think fraction, clock fraction); opponent plies are
             context, flagged as such
  residual   g(current board, z, ratings): a small transformer with no move history, added to the teacher's
             logits. Its output gain starts at 0, so step 0 is exactly Maia-3.
  think      censored likelihood of the think fraction think / (remaining + inc) on the logit scale: logistic
             main component plus a wide Cauchy outlier component (heavy tails); clocks are whole seconds, so the
             observation is the interval think ± 1 s
  resign     P(the side to move resigns in this position), weighted BCE

Gradient rule (memory "think-time-head"): only imitation losses exist in v0, so nothing outcome-driven can
reach the think head. Keep it that way: any future strength/RL loss must detach z and the head inputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from doppel_lab.v0_data import CLK_FEATS, ENC_FEATS, HIST, TEMPO_FEATS

Z = 8
NMOVES = 4352


class Trunk(nn.Module):
    """Frozen Maia-3. Returns teacher logits (N, 4352), value logits (N, 3), pooled features (N, dim)."""

    def __init__(self, maia_model: nn.Module):
        super().__init__()
        self.m = maia_model.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self._pooled = None
        self.m.last_ln.register_forward_hook(lambda mod, i, o: setattr(self, "_pooled", o))
        self.dim = self.m.last_ln.normalized_shape[0]

    def train(self, mode: bool = True):
        super().train(mode)
        self.m.eval()
        return self

    @torch.no_grad()
    def forward(self, tok, elo_self, elo_opp):
        logits, value, _ = self.m(tok, elo_self.float(), elo_opp.float())
        return logits.float(), value.float(), self._pooled.float()


class FakeTrunk(nn.Module):
    """For CPU tests: same interface, random fixed weights."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(64 * (12 * HIST + 1), dim)
        self.mv = nn.Linear(dim, NMOVES)
        self.val = nn.Linear(dim, 3)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, tok, elo_self, elo_opp):
        h = torch.tanh(self.lin(tok.flatten(1)))
        return self.mv(h), self.val(h), h


def gaussian(params):
    mu, logvar = params.chunk(2, -1)
    return mu, logvar.clamp(-8, 6)


class Prior(nn.Module):
    def __init__(self, cond_dim: int = 4, hid: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, hid), nn.GELU(), nn.Linear(hid, 2 * Z))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)  # starts at N(0, I)

    def forward(self, cond):
        return gaussian(self.net(cond))


class Encoder(nn.Module):
    def __init__(self, d: int = 128, layers: int = 2, heads: int = 4, cond_dim: int = 4):
        super().__init__()
        self.inp = nn.Linear(ENC_FEATS + cond_dim, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 2 * Z)

    def forward(self, enc, mask, cond):
        x = self.inp(torch.cat([enc, cond[:, None, :].expand(-1, enc.shape[1], -1)], -1))
        x = self.tf(x, src_key_padding_mask=~mask)
        own = mask & (enc[..., 0] > 0.5)
        w = own.float() / own.float().sum(1, keepdim=True).clamp(min=1)
        return gaussian(self.out((x * w[..., None]).sum(1)))


class Residual(nn.Module):
    """Per-move logit correction from the current board (no history), z and both ratings."""

    def __init__(self, d: int = 256, layers: int = 4, heads: int = 8, dh: int = 64):
        super().__init__()
        self.sq = nn.Parameter(torch.randn(64, d) * 0.02)
        self.piece = nn.Linear(12, d)
        self.ctx = nn.Linear(Z + 2, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.q_from, self.q_to = nn.Linear(d, dh, bias=False), nn.Linear(d, dh, bias=False)
        self.promo = nn.Linear(d, 4, bias=False)
        self.gain = nn.Parameter(torch.zeros(()))
        self.dh = dh
        r7, r8 = torch.arange(48, 56), torch.arange(56, 64)
        self.register_buffer("pf", r7.repeat_interleave(32), persistent=False)        # from square per promo
        self.register_buffer("pt", r8.repeat_interleave(4).repeat(8), persistent=False)  # to square per promo
        self.register_buffer("pp", torch.arange(4).repeat(64), persistent=False)       # piece per promo

    def board(self, tok):
        """Current-board embedding, shared across z samples."""
        cur = tok[:, :, 12 * (HIST - 1):12 * HIST]
        return self.piece(cur) + self.sq

    def forward(self, emb, z, elo_self, elo_opp):
        c = torch.cat([z, ((elo_self.float() - 1500) / 400)[:, None], ((elo_opp.float() - 1500) / 400)[:, None]], -1)
        x = self.tf(emb + self.ctx(c)[:, None, :])
        f, t = self.q_from(x), self.q_to(x)
        s = torch.einsum("bid,bjd->bij", f, t) / math.sqrt(self.dh)
        flat = s.reshape(len(x), 4096)
        promo = s[:, self.pf, self.pt] + self.promo(x[:, 56:64])[:, self.pt - 56, self.pp]
        return self.gain * torch.cat([flat, promo], 1)


class ThinkHead(nn.Module):
    def __init__(self, trunk_dim: int, hid: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(trunk_dim + CLK_FEATS + TEMPO_FEATS + Z, hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 2))
        self.outlier = nn.Parameter(torch.tensor(-3.0))  # logit of the outlier weight
        self.out_loc = nn.Parameter(torch.tensor(-2.0))
        self.out_scale = nn.Parameter(torch.tensor(math.log(3.0)))

    def forward(self, pooled, clk, tempo, z):
        mu, log_s = self.net(torch.cat([pooled, clk, tempo, z], -1)).unbind(-1)
        return mu, log_s.clamp(-4, 3)

    def nll(self, mu, log_s, think, before, inc):
        """Interval-censored NLL of think ∈ [think-1, think+1] s, as fraction of before + inc, logit scale."""
        budget = (before + inc).clamp(min=1.0)
        lo = ((think - 1.0).clamp(min=0.0) / budget).clamp(0.0, 1.0)
        hi = ((think + 1.0) / budget).clamp(0.0, 1.0)
        ylo, yhi = torch.logit(lo, eps=1e-6), torch.logit(hi, eps=1e-6)
        s = log_s.exp()
        main = torch.sigmoid((yhi - mu) / s) - torch.sigmoid((ylo - mu) / s)
        os_ = self.out_scale.exp()
        cauchy = (torch.atan((yhi - self.out_loc) / os_) - torch.atan((ylo - self.out_loc) / os_)) / math.pi
        pi = torch.sigmoid(self.outlier)
        return -torch.log((1 - pi) * main + pi * cauchy + 1e-9)

    @staticmethod
    def seconds(mu, before, inc):
        """Median think time of the main component, in seconds."""
        return torch.sigmoid(mu) * (before + inc).clamp(min=1.0)


class ResignHead(nn.Module):
    def __init__(self, trunk_dim: int, hid: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(trunk_dim + 3 + CLK_FEATS + Z, hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        nn.init.constant_(self.net[-1].bias, -5.0)  # base rate ~0.6%

    def forward(self, pooled, value, clk, z):
        return self.net(torch.cat([pooled, value.softmax(-1), clk, z], -1))[:, 0]


class DoppelV0(nn.Module):
    def __init__(self, trunk: nn.Module, n_prior_samples: int = 2):
        super().__init__()
        self.trunk = trunk
        self.prior, self.encoder, self.residual = Prior(), Encoder(), Residual()
        self.think, self.resign = ThinkHead(trunk.dim), ResignHead(trunk.dim)
        self.S = n_prior_samples

    def heads_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("trunk.") and p.requires_grad]


def legal_mask(legal: torch.Tensor) -> torch.Tensor:
    m = torch.zeros(len(legal), NMOVES + 1, dtype=torch.bool, device=legal.device)
    m.scatter_(1, torch.where(legal < 0, NMOVES, legal.long()), True)
    return m[:, :NMOVES]


def losses(model: DoppelV0, b: dict, beta: float, free_bits: float = 3.0, alpha: float = 0.5,
           w_think: float = 1.0, w_resign: float = 1.0, sample_post: bool = True) -> tuple[torch.Tensor, dict]:
    """One batch -> total loss and logged parts. Per-position scale: per-sample terms are divided by n_cand."""
    tok = b["tok"].float()
    t_logits, value, pooled = model.trunk(tok, b["elo_self"], b["elo_opp"])
    mask = legal_mask(b["legal"])
    t_logp = t_logits.masked_fill(~mask, -1e9).log_softmax(-1)

    mu_q, lv_q = model.encoder(b["enc"], b["enc_mask"], b["cond"])
    mu_p, lv_p = model.prior(b["cond"])
    z_s = mu_q + (torch.randn_like(mu_q) * (0.5 * lv_q).exp() if sample_post else 0)
    kl_d = 0.5 * (lv_p - lv_q + ((lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp()) - 1)  # (B, Z)
    kl = kl_d.sum(-1)
    kl_fb = kl_d.clamp(min=free_bits / Z).sum(-1)
    sid = b["sid"]
    z = z_s[sid]

    emb = model.residual.board(tok)
    s_logits = t_logp + model.residual(emb, z, b["elo_self"], b["elo_opp"])
    s_logp = s_logits.masked_fill(~mask, -1e9).log_softmax(-1)
    has_move = b["played"] >= 0
    pl_idx = b["played"].clamp(min=0)
    ce = -s_logp.gather(1, pl_idx[:, None])[:, 0]
    ce_t = -t_logp.gather(1, pl_idx[:, None])[:, 0]
    L_pol = ce[has_move].mean()

    # Marginal matching: the prior mixture of students should stay close to the teacher.
    mix = []
    std_p = (0.5 * lv_p).exp()
    for _ in range(model.S):
        zp = (mu_p + torch.randn_like(mu_p) * std_p)[sid]
        lg = t_logp + model.residual(emb, zp, b["elo_self"], b["elo_opp"])
        mix.append(lg.masked_fill(~mask, -1e9).log_softmax(-1))
    mix_logp = torch.logsumexp(torch.stack(mix), 0) - math.log(model.S)
    t_p = t_logp.exp()
    L_marg = (t_p * (t_logp - mix_logp)).masked_fill(~mask, 0).sum(-1)[has_move].mean()

    th = b["think"]
    ok = has_move & torch.isfinite(th[:, 0])
    mu_t, ls_t = model.think(pooled, b["clk"], b["tempo"], z)
    nll_t = model.think.nll(mu_t[ok], ls_t[ok], th[ok, 0], th[ok, 1], th[ok, 2])
    L_think = nll_t.mean() if ok.any() else torch.zeros((), device=tok.device)

    r_logit = model.resign(pooled, value, b["clk"], z)
    w = b["weight"]
    L_res = (w * F.binary_cross_entropy_with_logits(r_logit, b["resign"], reduction="none")).sum() / w.sum()

    n_pos = b["ncand"].sum()
    L_kl = kl_fb.sum() / n_pos
    total = L_pol + alpha * L_marg + beta * L_kl + w_think * L_think + w_resign * L_res
    with torch.no_grad():
        top1 = (s_logp.argmax(-1) == pl_idx)[has_move].float().mean()
        top1_t = (t_logp.argmax(-1) == pl_idx)[has_move].float().mean()
    parts = {"loss": total.item(), "ce": L_pol.item(), "ce_teacher": ce_t[has_move].mean().item(),
             "marg": L_marg.item(), "kl_nats": kl.mean().item(), "think_nll": L_think.item(),
             "resign_bce": L_res.item(), "top1": top1.item(), "top1_teacher": top1_t.item(),
             "gain": model.residual.gain.item()}
    return total, parts
