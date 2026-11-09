"""Do sampled v0 think times reproduce human slow/fast spells? (SCIENCE.md §5 "Long thinks cluster", open check)

    python -m doppel_lab.think_spells --run v0d --games 2000 --gpu-ok     # -> lab/results/think_spells_<run>.json

Humans: residual log think (after position difficulty, clock and the side's own pace in the game) correlates
+0.15 with the side's previous move's residual. v0's think head sees tempo_feats (median and 0.7-decay EWMA of the
side's recent think fractions), so it can carry spells, but only if its own sampled times are fed back.

Val games (held-out players), own moves at ply >= 10 with a labelled think. Real positions and clocks throughout
(the clock is not re-simulated, so time pressure follows the human's game). z drawn once per side-game from the
prior. Think fraction sampled from the head's mixture (logistic main component, Cauchy outlier), seconds =
fraction * (remaining + inc). Conditions:
  human      the recorded think times
  tf         sampled, tempo from the human's real previous thinks (teacher-forced: what training saw)
  ar         sampled, own tempo from the model's own previous samples (a Doppel playing a game)
  ar_median  as ar but each move takes the main component's median (no sampling noise)
Per condition: residual of log1p(think s) on [1, entropy, entropy², own clock fraction, log own clock, ply] pooled,
minus the side-game mean residual; then within side-game lag 1, 2, 3, 5 correlations over consecutive own moves.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from doppel_lab.v0_data import V0, Index, remaining, targets_sample, tempo_feats, think_fracs

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results"
LAGS = (1, 2, 3, 5)


def sample_frac(head, mu, log_s, gen):
    """Draw think fractions from the head's mixture on the logit scale."""
    u = torch.rand(mu.shape, generator=gen, device=mu.device).clamp(1e-6, 1 - 1e-6)
    main = mu + log_s.exp() * torch.log(u / (1 - u))
    v = torch.rand(mu.shape, generator=gen, device=mu.device)
    cau = head.out_loc + head.out_scale.exp() * torch.tan(math.pi * (v - 0.5))
    pick = torch.rand(mu.shape, generator=gen, device=mu.device) < torch.sigmoid(head.outlier)
    return torch.sigmoid(torch.where(pick, cau, main))


def lag_corr(r: np.ndarray, key: np.ndarray) -> dict:
    out = {}
    for L in LAGS:
        same = key[L:] == key[:-L]
        a, b = r[L:][same], r[:-L][same]
        out[str(L)] = float(np.corrcoef(a, b)[0, 1])
    return out


def residuals(y, X, key):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    _, k = np.unique(key, return_inverse=True)
    return r - (np.bincount(k, r) / np.bincount(k))[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0d")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--gpu-ok", action="store_true")
    a = ap.parse_args()
    device = "cuda" if a.gpu_ok else "cpu"
    from doppel_lab.v0_train import build
    model = build(False, device).eval()
    s = torch.load(V0 / "runs" / a.run / "ckpt.pt", map_location=device, weights_only=False)
    model.load_state_dict(s["model"], strict=False)
    ix = Index(V0 / "index_1500_val.npz")
    rng = np.random.default_rng(0)
    gen = torch.Generator(device=device).manual_seed(0)
    games = [g for g in rng.permutation(ix.G) if ix.plies(g) >= 40][:a.games]
    rows = {k: [] for k in ("key", "ply", "ent", "own_frac", "log_own", "human", "tf", "ar", "ar_median")}
    t0 = time.time()
    for gi, g in enumerate(games):
        frac_h = think_fracs(ix, g)
        a0, n = int(ix.off[g]), ix.plies(g)
        clock = ix.clock[a0:a0 + n]
        base, inc = float(ix.base[g]), float(ix.inc[g])
        for color in (0, 1):
            cand = [k for k in range(color, n, 2) if k >= 10 and np.isfinite(frac_h[k])]
            if len(cand) < 8:
                continue
            b = targets_sample(ix, g, color, cand, None, len(cand), rng, frac_h)
            if b is None:
                continue
            T = len(b["ply"])
            tok = torch.from_numpy(b["tok"]).to(device).float()
            es, eo = torch.from_numpy(b["elo_self"]).to(device), torch.from_numpy(b["elo_opp"]).to(device)
            with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
                _, _, pooled = model.trunk(tok, es, eo)
            pooled = pooled.float()
            with torch.no_grad():
                pm = model.prior(torch.from_numpy(b["cond"][None]).to(device))
                mu_z, logvar_z = pm
                z = (mu_z + (0.5 * logvar_z).exp() * torch.randn(mu_z.shape, generator=gen, device=device)).expand(T, -1)
                clk = torch.from_numpy(b["clk"]).to(device)
                budget = torch.from_numpy(b["think"][:, 1] + b["think"][:, 2]).to(device).clamp(min=1.0)
                mu, ls = model.think(pooled, clk, torch.from_numpy(b["tempo"]).to(device), z)
                tf = sample_frac(model.think, mu, ls, gen) * budget
                # autoregressive: own tempo from own samples; opponent tempo stays the human's
                ar, arm = np.zeros(T), np.zeros(T)
                own_s, own_m = [], []
                for i, k in enumerate(b["ply"].tolist()):
                    prev_opp = [frac_h[j] for j in range(1 - color, k, 2) if np.isfinite(frac_h[j])]
                    tmp = torch.tensor([tempo_feats(own_s, prev_opp), tempo_feats(own_m, prev_opp)],
                                       dtype=torch.float32, device=device)
                    m2, l2 = model.think(pooled[i:i + 1].expand(2, -1), clk[i:i + 1].expand(2, -1), tmp, z[:2])
                    bud = float(budget[i])
                    fs = float(sample_frac(model.think, m2[:1], l2[:1], gen)[0])
                    fm = float(torch.sigmoid(m2[1]))
                    ar[i], arm[i] = fs * bud, fm * bud
                    own_s.append((ar[i] + 0.5) / (bud + 1.0))
                    own_m.append((arm[i] + 0.5) / (bud + 1.0))
            own = np.array([remaining(clock, k, base)[0] for k in b["ply"].tolist()])
            rows["key"] += [g * 2 + color] * T
            rows["ply"] += b["ply"].tolist()
            rows["ent"] += ix.entropy[a0 + b["ply"]].astype(float).tolist()
            rows["own_frac"] += (own / max(base, 1.0)).tolist()
            rows["log_own"] += np.log1p(own).tolist()
            rows["human"] += b["think"][:, 0].tolist()
            rows["tf"] += tf.cpu().numpy().tolist()
            rows["ar"] += ar.tolist()
            rows["ar_median"] += arm.tolist()
        if (gi + 1) % 200 == 0:
            print(f"{gi + 1}/{len(games)} games, {len(rows['key']):,} moves ({time.time() - t0:.0f}s)", flush=True)
    R = {k: np.array(v, float) for k, v in rows.items()}
    key = R["key"].astype(np.int64)
    X = np.c_[np.ones(len(key)), R["ent"], R["ent"] ** 2, R["own_frac"], R["log_own"], np.minimum(R["ply"], 120)]
    out = {"run": a.run, "games": len(games), "moves": int(len(key)), "conditions": {}}
    yh = np.log1p(R["human"])
    for cnd in ("human", "tf", "ar", "ar_median"):
        y = np.log1p(np.maximum(R[cnd], 0))
        r = residuals(y, X, key)
        out["conditions"][cnd] = {"lag_corr": lag_corr(r, key), "mean_log_s": float(y.mean()),
                                  "sd_log_s": float(y.std()), "r_log_with_human": float(np.corrcoef(y, yh)[0, 1]),
                                  "sd_within_side_game": float(r.std())}
        print(cnd, json.dumps(out["conditions"][cnd]), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"think_spells_{a.run}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"-> think_spells_{a.run}.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
