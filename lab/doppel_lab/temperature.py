"""Sampling temperature vs human blunder rate: does Maia-3 at 1500 over-blunder when sampled, and what fixes it? (CPU)

    python -m doppel_lab.temperature [where|truncation|elo]    # needs punish.py --random (writes the per-position table)

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



def truncation():
    """Top-p and min-p: implied blunder rate, entropy, and how often the human's move is cut from the support."""
    d = pl.read_parquet(V0 / "policy_implied_positions_1500_val.parquet")
    data = []
    for p, b, m, u in zip(d["p"].to_list(), d["is_blunder"].to_list(), d["moves"].to_list(), d["played"].to_list()):
        p, b = np.asarray(p, float), np.asarray(b, bool)
        if p.sum() > 0 and u in m:
            data.append((p / p.sum(), b, m.index(u)))
    human = float(np.mean([b[i] for _, b, i in data]))
    rules = [("none", None)] + [(f"top_p {x}", ("top", x)) for x in (0.99, 0.97, 0.95, 0.9)] + \
            [(f"min_p {x}", ("min", x)) for x in (0.005, 0.01, 0.02, 0.05)]
    res = {"human_fresh_blunder": human, "rules": {}}
    print(f"{len(data):,} positions; human (fresh flags) {human:.4f}")
    for name, r in rules:
        bl, ent, cut, top = [], [], [], []
        for p, b, i in data:
            keep = np.ones(len(p), bool)
            if r and r[0] == "top":
                o = np.argsort(-p)
                cs = np.cumsum(p[o])
                k = int(np.searchsorted(cs, r[1]) + 1)
                keep = np.zeros(len(p), bool)
                keep[o[:k]] = True
            elif r and r[0] == "min":
                keep = p >= r[1] * p.max()
            q = np.where(keep, p, 0)
            q /= q.sum()
            bl.append(q[b].sum()); ent.append(-(q[q > 0] * np.log(q[q > 0])).sum()); cut.append(not keep[i]); top.append(q.max())
        res["rules"][name] = {"implied_blunder": float(np.mean(bl)), "entropy": float(np.mean(ent)),
                              "human_move_cut": float(np.mean(cut)), "p_top": float(np.mean(top))}
        v = res["rules"][name]
        print(f"{name:12s} implied blunder {v['implied_blunder']:.4f} ({v['implied_blunder'] / human:.2f}x)  entropy "
              f"{v['entropy']:.3f}  p_top {v['p_top']:.3f}  human move cut {v['human_move_cut']:.4f}", flush=True)
    (RESULTS / "truncation.json").write_text(json.dumps(res, indent=1) + "\n")



def elo_offset(elos=(1300, 1500, 1700, 1900, 2100)):
    """Maia-3's rating input as the knob: implied blunder rate on the same positions at several self/opp ratings."""
    from doppel_lab.engines import Maia3
    from doppel_lab.punish import maia_probs
    d = pl.read_parquet(V0 / "policy_implied_positions_1500_val.parquet")
    games = dict(pl.read_parquet(V0 / "blunder_games_1500_val.parquet", columns=["game_id", "moves"]).iter_rows())
    model = Maia3("79m", "cpu").model.eval()
    flags = [dict(zip(m, b)) for m, b in zip(d["moves"].to_list(), d["is_blunder"].to_list())]
    human = float(np.mean([f[u] for f, u in zip(flags, d["played"].to_list()) if u in f]))
    res = {"human_fresh_blunder": human, "elos": {}}
    for e in elos:
        pol = maia_probs(d, games, e, model)
        bl = [sum(p for u, p in pm.items() if f.get(u, False)) for pm, f in zip(pol, flags)]
        top = [max(pm.values()) for pm in pol]
        ll = [np.log(max(pm.get(u, 1e-12), 1e-12)) for pm, u in zip(pol, d["played"].to_list())]
        res["elos"][str(e)] = {"implied_blunder": float(np.mean(bl)), "p_top": float(np.mean(top)),
                               "human_move_ll": float(np.mean(ll))}
        print(f"Maia-3 at {e}: implied blunder {np.mean(bl):.4f} ({np.mean(bl) / human:.2f}x human)  p_top {np.mean(top):.3f}  "
              f"human-move log-lik {np.mean(ll):.4f}", flush=True)
    (RESULTS / "temperature_elo.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    import sys
    {"where": where, "truncation": truncation, "elo": elo_offset}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
