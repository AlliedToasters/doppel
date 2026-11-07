"""Encoder benchmark, Doppel side: the v0 z encoder's posterior means on the bench games (CPU).

    python -m doppel_lab.bench_doppel --run v0d          # needs data/bench/maia3.npz (per-position policy arrays)

The encoder reads one colour's context half (v0_data.build_sample): per ply own/opponent flag, the played move's
teacher log-prob, its top-16 rank, the teacher entropy, think fraction (logit) and its validity flag, own clock
over base, ply/100. Teacher features come from bench_embed (Maia-3 79M at 1500 vs 1500). The rating condition
is fixed at 1500 vs 1500 as well, so the encoder is rating-blind like every other bench model; base and
increment stay, since the clocks carry them anyway.
Representation per game: mu for (White, Black) x (first half, second half) = 32 dims, plus the per-colour mean.
The v0 encoder was trained on the 1500 band only (2026-10), so this is an out-of-band test for other ratings.
Probes as bench_embed: linear (standardised, ridge/logistic) and gradient-boosted trees, alone and with surface.
Output: lab/results/bench_doppel_<run>.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import polars as pl
import torch

from doppel_lab.bench import SEED, TASKS, BENCH
from doppel_lab.v0_data import _logit, remaining, think_seconds
from doppel_lab.v0_model import Encoder

ROOT = BENCH.parents[1]
RESULTS = ROOT / "lab" / "results"


def tokens(lp, rank, ent, clock, base, inc, color, ctx):
    n = len(lp)
    frac = np.full(n, np.nan)
    for k in range(n):
        t, before = think_seconds(clock, k, base, inc)
        if np.isfinite(t):
            frac[k] = (t + 0.5) / (before + inc + 1.0)
    enc = []
    for k in ctx:
        own, _ = remaining(clock, k, base)
        f = frac[k]
        enc.append([float((k % 2) == color), float(lp[k]), rank[k] / 16.0, float(ent[k]),
                    float(_logit(f)) if np.isfinite(f) else 0.0, float(np.isfinite(f)), own / max(base, 1.0), k / 100])
    return np.array(enc, np.float32)


def embed(run: str, suite: str = "lichess") -> tuple[np.ndarray, np.ndarray]:
    from doppel_lab.bench import suite_dir
    D = suite_dir(suite)
    s = torch.load(ROOT / "data" / "v0" / "runs" / run / "ckpt.pt", map_location="cpu", weights_only=False)
    enc = Encoder()
    enc.load_state_dict({k[len("encoder."):]: v for k, v in s["model"].items() if k.startswith("encoder.")})
    enc.eval()
    z = np.load(D / "maia3.npz", allow_pickle=True)
    off = z["pos_off"]
    g = pl.read_parquet(D / "games.parquet", columns=["game_id", "clocks", "base", "inc"])
    assert (g["game_id"].to_numpy() == z["game_ids"]).all()
    G = g.height
    mu = np.zeros((G, 2, 2, 8), np.float32)
    batch, meta = [], []

    def flush():
        L = max(len(e) for e, _ in batch)
        x = np.zeros((len(batch), L, 8), np.float32)
        m = np.zeros((len(batch), L), bool)
        cond = np.zeros((len(batch), 4), np.float32)
        for i, (e, cd) in enumerate(batch):
            x[i, :len(e)], m[i, :len(e)], cond[i] = e, True, cd
        with torch.no_grad():
            out, _ = enc(torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(cond))
        for (gi, col, half), v in zip(meta, out.numpy()):
            mu[gi, col, half] = v
        batch.clear()
        meta.clear()

    for gi, (clocks, base, inc) in enumerate(zip(g["clocks"].to_list(), g["base"].to_list(), g["inc"].to_list())):
        a, b = off[gi], off[gi + 1]
        n = b - a
        lp, rank, ent = z["pos_lp"][a:b].astype(np.float32), z["pos_rank"][a:b], z["pos_ent"][a:b].astype(np.float32)
        clock = np.asarray(clocks[:n], np.float32)
        cd = np.array([0.0, 0.0, math.log1p(base), inc / max(base, 1.0)], np.float32)   # ratings fixed at 1500
        mid = n // 2
        for col in (0, 1):
            for half, ctx in ((0, range(0, mid)), (1, range(mid, n))):
                e = tokens(lp, rank, ent, clock, float(base), float(inc), col, ctx)
                if len(e) and e[:, 0].any():
                    batch.append((e, cd))
                    meta.append((gi, col, half))
        if len(batch) >= 2048:
            flush()
        if gi % 20000 == 0:
            print(f"{gi:,}/{G:,}", flush=True)
    if batch:
        flush()
    return z["game_ids"], mu


def probe(run: str):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    gid, mu = embed(run)
    Z = np.concatenate([mu.reshape(len(gid), -1), mu.mean(2).reshape(len(gid), -1)], 1)
    ix = pl.DataFrame({"game_id": gid, "i": np.arange(len(gid))})
    d = pl.read_parquet(BENCH / "tasks.parquet").join(ix, on="game_id").join(pl.read_parquet(BENCH / "surface.parquet"),
                                                                             on="game_id")
    scols = [k for k in pl.read_parquet(BENCH / "surface.parquet").columns if k != "game_id"]
    surface = json.loads((RESULTS / "bench_surface.json").read_text())["tasks"]
    res = {"run": run, "tasks": {}}
    for name, (flt, lab, kind) in TASKS.items():
        x = d.filter(flt).with_columns(y=lab)
        y, fold, ii = x["y"].to_numpy().astype(float), x["fold"].to_numpy(), x["i"].to_numpy()
        Xs = x.select(scols).to_numpy().astype(float)
        sets = {"z_linear": ("lin", Z[ii]), "z_hgb": ("hgb", Z[ii]), "surface+z": ("hgb", np.concatenate([Xs, Z[ii]], 1))}
        r = {"surface": surface[name]}
        for sname, (how, X) in sets.items():
            pred = np.empty(len(y))
            for k in range(5):
                tr, te = fold != k, fold == k
                if how == "lin":
                    sc = StandardScaler().fit(X[tr])
                    A, B = sc.transform(X[tr]), sc.transform(X[te])
                    pred[te] = (Ridge(alpha=1.0).fit(A, y[tr]).predict(B) if kind == "reg"
                                else LogisticRegression(max_iter=2000).fit(A, y[tr]).predict_proba(B)[:, 1])
                else:
                    M = HistGradientBoostingRegressor if kind == "reg" else HistGradientBoostingClassifier
                    m = M(max_iter=400, learning_rate=0.05, random_state=SEED).fit(X[tr], y[tr])
                    pred[te] = m.predict(X[te]) if kind == "reg" else m.predict_proba(X[te])[:, 1]
            r[sname] = ({"mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))} if kind == "reg"
                        else {"auc": float(roc_auc_score(y, pred))})
        res["tasks"][name] = r
        key = "r2" if kind == "reg" else "auc"
        print(f"{name:13s} " + "  ".join(f"{k} {v[key]:.3f}" for k, v in r.items()), flush=True)
    (RESULTS / f"bench_doppel_{run}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0d")
    probe(ap.parse_args().run)
