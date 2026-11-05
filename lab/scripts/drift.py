"""Phase 1 steps 2-3: is per-game error persistence real state, or position difficulty? Does it move after a blunder?

Step 2 (difficulty adjustment). Each move's y = log1p(CPL) is residualised on its position context
(mover-POV eval bin x ply bin x rating band [x clock-fraction bin, humans only]); cell means are fitted
on human moves. The within-player test (scripts/within_player.py) is rerun on residual half-means, and the
shared-environment correlation r(my h1, opponent h2) is reported as the part a difficulty story explains.

Step 3 (post-blunder drift). For each player-game whose first blunder is at own move t (t >= 4, >= 10 own
moves after), delta_k = resid_y[t+k] - mean(resid_y[:t]) for k = 1..10, minus the same quantity in control
player-games with no blunder through t (same source, same t). Run on human movers, on the human opponent
after the blunder (placebo: shared position), and on Maia-3 / ChessMimic self-play (no state; a position-only null).
Fixed per-game z predicts no post-blunder shift; time-varying z_t predicts a rise that may decay.

Run from repo root: PYTHONPATH=. .venv/bin/python lab/scripts/drift.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.metrics import BLUNDER, CLIP, adjudicate, wc

rng = np.random.default_rng(0)
K, T_MIN, MIN_GAMES = 10, 4, 8
EVAL_BINS = np.array([-400, -150, -50, 50, 150, 400])
CLOCK_BINS = np.array([0.1, 0.25, 0.5])


def load_human():
    g = pl.concat([pl.read_parquet("data/human/eval_2026-10.parquet"), pl.read_parquet("data/human/repeat_2026-10.parquet")],
                  how="diagonal_relaxed").unique("game_id")
    ev = pl.concat([pl.read_parquet(f) for f in ("data/evals/human.parquet", "data/evals/human_repeat.parquet")]).unique("game_id")
    return g.join(ev, on="game_id"), "human"


def load_synth(engine):
    out = []
    for b in (1100, 1300, 1500, 1700, 1900, 2100):
        g = pl.read_parquet(f"data/synth/{engine}_{b}.parquet").drop("clocks")
        out.append(g.join(pl.read_parquet(f"data/evals/{engine}_{b}.parquet"), on="game_id"))
    return pl.concat(out, how="diagonal_relaxed"), engine


def moves_table(games, source):
    """One row per adjudicated move, with context cells."""
    recs = []
    for r in games.iter_rows(named=True):
        ev = np.clip(np.asarray(r["evals"], float), -CLIP, CLIP)
        end = adjudicate(ev)
        if end < 12:
            continue
        sign = np.where(np.arange(end) % 2 == 0, 1.0, -1.0)
        before, after = ev[:end] * sign, ev[1:end + 1] * sign
        cpl = np.maximum(0.0, before - after)
        bl = (wc(before) - wc(after)) >= BLUNDER
        clocks = r.get("clocks")
        base = max(1.0, float(r["base"] or 1))
        for k in range(end):
            side = k % 2
            name = (r["white"] if side == 0 else r["black"]).lower() if source == "human" else f"{source}:{r['game_id']}:{side}"
            cf = clocks[k] / base if clocks is not None and k < len(clocks) and clocks[k] is not None else np.nan
            recs.append((r["game_id"], side, k // 2, name, r["band"], int(np.digitize(before[k], EVAL_BINS)),
                         min(k // 10, 8), -1 if np.isnan(cf) else int(np.digitize(cf, CLOCK_BINS)),
                         float(np.log1p(cpl[k])), bool(bl[k])))
    return pl.DataFrame(recs, orient="row", schema=["game_id", "side", "i", "player", "band", "ebin", "pbin", "cbin", "y", "bl"])


def residualise(df, cells_ref, keys):
    """Subtract cell means fitted on the human reference; fall back to coarser cells when sparse."""
    out = df
    fine = cells_ref.group_by(keys).agg(pl.col("y").mean().alias("m_fine"), pl.len().alias("n_fine"))
    coarse = cells_ref.group_by(["ebin", "pbin"]).agg(pl.col("y").mean().alias("m_coarse"))
    out = out.join(fine, on=keys, how="left").join(coarse, on=["ebin", "pbin"], how="left")
    m = pl.when(pl.col("n_fine") >= 30).then(pl.col("m_fine")).otherwise(pl.col("m_coarse"))
    return out.with_columns((pl.col("y") - m.fill_null(pl.col("y").mean())).alias("r")).drop("m_fine", "n_fine", "m_coarse")


# ---------- step 2: within-player test on residuals ----------
def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def within_player(df, col):
    halves = (df.with_columns((pl.col("i") >= (pl.col("i").max().over(["game_id", "side"]) + 1) // 2).alias("h2"))
                .group_by(["game_id", "side", "player", "h2"]).agg(pl.col(col).mean().alias("v"))
                .pivot(on="h2", index=["game_id", "side", "player"], values="v")
                .rename({"false": "h1", "true": "h2"}).drop_nulls())
    opp = halves.select("game_id", (1 - pl.col("side")).alias("side"), pl.col("h2").alias("opp_h2"))
    halves = halves.join(opp, on=["game_id", "side"], how="left")
    keep = halves.group_by("player").len().filter(pl.col("len") >= MIN_GAMES)["player"]
    P = {k: g.select("h1", "h2", "opp_h2").to_numpy() for k, g in
         halves.filter(pl.col("player").is_in(keep.implode())).group_by("player")}
    P = {k[0]: v for k, v in P.items()}

    def stats(keys):
        wg, cg, res, env = [], [], [], []
        for k in keys:
            X = P[k]
            x, y, o = X[:, 0], X[:, 1], X[:, 2]
            n = len(x)
            wg.append(np.c_[x, y]); cg.append(np.c_[x, np.roll(y, 1)])
            res.append(np.c_[x - (x.sum() - x) / (n - 1), y - (y.sum() - y) / (n - 1)])
            ok = ~np.isnan(o); env.append(np.c_[x[ok], o[ok]])
        f = lambda L: corr(*np.concatenate(L).T)
        return f(wg), f(cg), f(res), f(env)

    keys = list(P)
    point = stats(keys)
    boots = np.array([stats(rng.choice(keys, len(keys))) for _ in range(200)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    share = (boots[:, 0] - boots[:, 1]) / boots[:, 0]
    names = ("within_game", "cross_game", "residual", "shared_env_opp")
    return {n: {"r": round(point[i], 4), "ci": [round(lo[i], 4), round(hi[i], 4)]} for i, n in enumerate(names)} | {
        "per_game_share": [round(float(np.percentile(share, q)), 3) for q in (2.5, 50, 97.5)],
        "n_players": len(keys), "n_player_games": int(sum(len(v) for v in P.values()))}


# ---------- step 3: post-blunder drift ----------
def drift(df, who="self"):
    seqs = {k: (g.sort("i")["r"].to_numpy(), g.sort("i")["bl"].to_numpy())
            for k, g in df.group_by(["game_id", "side"])}
    first = {k: (int(np.argmax(b)) if b.any() else None) for k, (r, b) in seqs.items()}
    # pre = r[:t]; post = r[t+off : t+off+K]. Self: off=1 (skip the blunder itself). Opponent: off=0.
    off = 1 if who == "self" else 0
    ctrl = defaultdict(list)  # player-games with no own blunder through t+off-1
    for k, (r, b) in seqs.items():
        f = first[k]
        for t in range(T_MIN, len(r) - K - off + 1):
            if f is not None and f < t + off:
                break
            ctrl[t].append(r[t + off:t + off + K] - r[:t].mean())
    ctrl = {t: np.mean(v, axis=0) for t, v in ctrl.items() if len(v) >= 20}
    ev = []
    for (gid, side), f in first.items():
        if f is None or f < T_MIN:
            continue
        if who == "self":
            r, t = seqs[(gid, side)][0], f
        else:  # White blunders at ply 2f: Black's own move f answers it. Black at 2f+1: White's own move f+1.
            if (gid, 1 - side) not in seqs:
                continue
            r, t = seqs[(gid, 1 - side)][0], (f if side == 0 else f + 1)
        if len(r) < t + off + K or t not in ctrl:
            continue
        ev.append(r[t + off:t + off + K] - r[:t].mean() - ctrl[t])
    ev = np.array(ev)
    boots = np.array([ev[rng.integers(0, len(ev), len(ev))].mean(axis=0) for _ in range(300)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"n_events": len(ev), "delta_by_k": np.round(ev.mean(axis=0), 4).tolist(),
            "ci_lo": np.round(lo, 4).tolist(), "ci_hi": np.round(hi, 4).tolist(),
            "mean_k1_5": round(float(ev[:, :5].mean()), 4), "mean_k6_10": round(float(ev[:, 5:].mean()), 4)}


if __name__ == "__main__":
    hg, _ = load_human()
    H = moves_table(hg, "human")
    print(f"human moves: {H.height}")
    out = {"human": {}}
    H_noclock = residualise(H, H, ["ebin", "pbin", "band"])
    H_clock = residualise(H, H, ["ebin", "pbin", "band", "cbin"])
    out["human"]["within_raw"] = within_player(H, "y")
    out["human"]["within_resid_position"] = within_player(H_noclock, "r")
    out["human"]["within_resid_position_clock"] = within_player(H_clock, "r")
    for k in ("within_raw", "within_resid_position", "within_resid_position_clock"):
        print(k, json.dumps(out["human"][k]))
    out["human"]["drift_self"] = drift(H_noclock, "self")
    out["human"]["drift_opponent_placebo"] = drift(H_noclock, "opponent")
    out["human"]["drift_self_clockadj"] = drift(H_clock, "self")
    for k in ("drift_self", "drift_opponent_placebo", "drift_self_clockadj"):
        print(k, json.dumps(out["human"][k]))
    for eng in ("maia3", "chessmimic"):
        sg, _ = load_synth(eng)
        S = residualise(moves_table(sg, eng), H, ["ebin", "pbin", "band"])
        out[eng] = {"drift_self": drift(S, "self"), "drift_opponent_placebo": drift(S, "opponent")}
        print(eng, json.dumps(out[eng]))
    Path("lab/results").mkdir(exist_ok=True)
    json.dump(out, open("lab/results/drift_2026-10.json", "w"), indent=1)
    print("wrote lab/results/drift_2026-10.json")
