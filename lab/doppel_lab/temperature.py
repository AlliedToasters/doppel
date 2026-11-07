"""Sampling temperature vs human blunder rate: does Maia-3 at 1500 over-blunder when sampled, and what fixes it? (CPU)

    python -m doppel_lab.temperature [where]    # needs punish.py --random (writes the per-position table)

6,000 random val positions (1500 band) with Maia-3 probabilities over legal moves and per-move blunder flags
(Stockfish 50k nodes, the label rule). For temperature T, p_T ∝ p^(1/T); implied blunder rate = mean over
positions of the p_T mass on blundering moves. Also the implied rate of playing Maia's top move (a proxy for
the move-match ceiling) and the entropy. Compared with the humans' actual blunder rate in the same positions.
Output: lab/results/temperature.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from doppel_lab.punish import RESULTS, V0

TS = (1.2, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.5)


def main():
    d = pl.read_parquet(V0 / "policy_implied_positions_1500_val.parquet")
    human = float(d["human_blunder"].cast(pl.Float64).mean())
    P = [np.asarray(p, float) for p in d["p"].to_list()]
    B = [np.asarray(b, bool) for b in d["is_blunder"].to_list()]
    res = {"positions": d.height, "human_blunder": human, "by_T": {}}
    # label agreement: the fresh per-move flag of the move actually played vs the stored human label
    fresh = np.array([b[m.index(u)] if u in m else np.nan for b, m, u in
                      zip(B, d["moves"].to_list(), d["played"].to_list())], float)
    lab = d["human_blunder"].cast(pl.Float64).to_numpy()
    ok = np.isfinite(fresh)
    res["played_fresh_blunder"] = float(fresh[ok].mean())
    res["label_agreement"] = float((fresh[ok] == lab[ok]).mean())
    print(f"{d.height:,} positions; human blunder rate {human:.4f} (stored labels); the played move re-flagged fresh "
          f"{fresh[ok].mean():.4f}; agreement {res['label_agreement']:.3f}")
    for T in TS:
        bl, top, ent = [], [], []
        for p, b in zip(P, B):
            if p.sum() <= 0:
                continue
            q = np.power(np.maximum(p, 1e-12), 1 / T)
            q /= q.sum()
            bl.append(q[b].sum())
            top.append(q.max())
            ent.append(-(q * np.log(q + 1e-12)).sum())
        r = {"implied_blunder": float(np.mean(bl)), "p_top": float(np.mean(top)), "entropy": float(np.mean(ent))}
        res["by_T"][str(T)] = r
        print(f"T {T:4.2f}: implied blunder {r['implied_blunder']:.4f} ({r['implied_blunder'] / human:.2f}x human)  "
              f"mean p_top {r['p_top']:.3f}  entropy {r['entropy']:.3f}", flush=True)
    (RESULTS / "temperature.json").write_text(json.dumps(res, indent=1) + "\n")


def where():
    """Where the T = 1 over-blunder lives: implied vs human (fresh flag) by blunder-potential and Maia-top bins."""
    d = pl.read_parquet(V0 / "policy_implied_positions_1500_val.parquet")
    rows = []
    for p, b, m, u in zip(d["p"].to_list(), d["is_blunder"].to_list(), d["moves"].to_list(), d["played"].to_list()):
        p, b = np.asarray(p, float), np.asarray(b, bool)
        if p.sum() <= 0 or u not in m:
            continue
        p = p / p.sum()
        rows.append((b.mean(), p.max(), p[b].sum(), float(b[m.index(u)]), bool(b[p.argmax()])))
    x = pl.DataFrame(rows, orient="row", schema=["beta", "p_top", "implied", "human", "top_is_blunder"])
    x = x.with_columns(bb=pl.col("beta").cut([0.25, 0.5, 0.75, 0.9], labels=["<25%", "25-50%", "50-75%", "75-90%", ">90%"]),
                       tb=pl.col("p_top").cut([0.3, 0.5, 0.7, 0.9], labels=["<.3", ".3-.5", ".5-.7", ".7-.9", ">.9"]))
    out = {}
    for k in ("bb", "tb"):
        t = x.group_by(k).agg(n=pl.len(), human=pl.col("human").mean(), implied=pl.col("implied").mean(),
                              top_blunders=pl.col("top_is_blunder").cast(pl.Float64).mean()).sort(k)
        out[k] = t.to_dicts()
        with pl.Config(float_precision=4):
            print(t)
    (RESULTS / "temperature_where.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    import sys
    where() if len(sys.argv) > 1 and sys.argv[1] == "where" else main()
