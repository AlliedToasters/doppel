"""Per-player empirical next-move distributions at repeated positions: how well does Maia fit each player? (CPU)

    python -m doppel_lab.player_markov coverage    # -> lab/results/player_markov.json "coverage"
    python -m doppel_lab.player_markov score       # -> lab/results/player_markov.json "score"

The user's idea (banked 2026-11-08): players are consistent at repeated board states (the position book's "the
position, not the path"), so each player has a sparse Markov-chain-like dataset: what they actually played at
positions they reached more than once. Scoring a model against that signal asks "how good a job is Maia doing
for this player?" directly, rather than assuming Maia is the right prior (the KL-to-Maia objective does).

Data: the 700k v0 1500-band train games. Position key = Polyglot Zobrist of the decision position (poskey,
R0, so transpositions merge), plies 0-79; ply 0 is the start position. Maia = the stored v0 teacher (Maia-3 79M
at the players' real Elos; log-prob of the played move). Visits are ordered by game time; a repeat of the same
position inside one game counts once (the first visit).

score: for every visit with at least one earlier visit by the same player, the played move's log-likelihood
under (a) Maia alone and (b) the player's own earlier moves there, shrunk toward Maia:
    p = (n_prev(move) + alpha * p_maia(move)) / (N_prev + alpha)
Online (only earlier games), so it is a fair forecast. The gain (b) - (a) in nats per visit, pooled and per player,
by N_prev, ply and popularity, measures how far real players sit from Maia where their own data can show it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.book import move_ids
from doppel_lab.posbook_common import annotate, ply_index

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "data/v0"
RESULTS = ROOT / "lab/results/player_markov.json"
CAP = 80
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
c = pl.col


def visits() -> pl.DataFrame:
    """One row per (player, game, decision position): player, t, key, mv, lp_maia, ply."""
    t0 = time.time()
    games = V0 / "games_1500_train.parquet"
    g = pl.read_parquet(games, columns=["game_id", "moves"])
    keys, lens = annotate("v0_1500_train_cap80", [games], cap=CAP)
    keys = np.asarray(keys)
    meta = np.load(ROOT / "data/v0e/meta_1500_train.npz", allow_pickle=True)
    assert (meta["game_id"] == g["game_id"].to_numpy()).all()
    gi, ply = ply_index(lens)
    off = np.r_[0, np.cumsum(lens)]
    # decision key at ply k = key after ply k-1; ply 0 = the start position (sentinel 0)
    dec = np.empty_like(keys)
    dec[1:] = keys[:-1]
    dec[off[:-1][lens > 0]] = 0
    ids, moff = move_ids(g["moves"])
    mv = ids[moff[gi] + ply]
    lp = np.empty(len(gi), np.float32)
    pos = 0
    for s in sorted((V0 / "teacher_1500_train").glob("shard-*.npz")):
        z = np.load(s)
        so = z["offsets"]
        for j in range(len(z["game_ids"])):  # shards are in game order; keep the first CAP plies of each
            n = lens[pos]
            lp[off[pos]:off[pos] + n] = z["played_lp"][so[j]:so[j] + n]
            pos += 1
    assert pos == len(lens)
    side = (ply % 2).astype(np.int64)
    df = pl.DataFrame({"player": meta["pid"][gi, side].astype(np.int64), "t": meta["t"][gi].astype(np.int64),
                       "g": gi.astype(np.int32), "key": dec.view(np.int64) if dec.dtype == np.uint64 else dec,
                       "mv": mv, "lp": lp, "ply": ply.astype(np.int16)})
    df = df.unique(["player", "g", "key"], keep="first", maintain_order=True)
    print(f"{df.height:,} visits from {len(lens):,} games ({time.time() - t0:.0f}s)", flush=True)
    return df


def online(df: pl.DataFrame) -> pl.DataFrame:
    """Earlier visits by the same player to the same position: N_prev, and n_prev of the move played now."""
    return (df.sort("player", "key", "t", "g")
            .with_columns(N_prev=pl.int_range(pl.len()).over("player", "key"),
                          n_prev=pl.int_range(pl.len()).over("player", "key", "mv"),
                          pos_total=pl.len().over("key")))


def coverage(df: pl.DataFrame) -> dict:
    x = online(df)
    rep = x.filter(c("N_prev") >= 1)
    per = rep.group_by("player").len()
    allp = df.select(c("player").n_unique()).item()
    out = {"visits": df.height, "repeat_visits": rep.height, "repeat_share": rep.height / df.height,
           "players": allp, "players_with_repeats": per.height,
           "players_by_min_repeats": {str(k): int((per["len"] >= k).sum()) for k in (1, 10, 30, 100, 300)},
           "repeat_ply_quantiles": {str(q): float(rep["ply"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "repeat_share_by_N_prev": {str(k): int((rep["N_prev"] == k).sum()) for k in range(1, 6)},
           "repeat_visits_ply_ge_10": int((rep["ply"] >= 10).sum())}
    print(json.dumps(out, indent=1), flush=True)
    return out


def score(df: pl.DataFrame) -> dict:
    x = online(df).filter(c("N_prev") >= 1)
    pm = np.exp(x["lp"].to_numpy().astype(np.float64))
    n, N = x["n_prev"].to_numpy().astype(np.float64), x["N_prev"].to_numpy().astype(np.float64)
    lm = np.log(np.clip(pm, 1e-9, 1))
    gains = {a: np.log((n + a * pm) / (N + a) + 1e-12) - lm for a in ALPHAS}
    pooled = {str(a): float(v.mean()) for a, v in gains.items()}
    best = max(ALPHAS, key=lambda a: gains[a].mean())
    gn = gains[best]
    x = x.with_columns(gain=pl.Series(gn), lm=pl.Series(lm), hit=pl.Series(n > 0))
    out = {"visits_scored": x.height, "alpha_grid_mean_gain": pooled, "alpha_best": best,
           "gain_nats_per_visit": float(gn.mean()), "maia_ll_per_visit": float(lm.mean()),
           "seen_before_share": float((n > 0).mean()),
           "maia_p_of_repeat_moves": float(pm[n > 0].mean()), "maia_p_of_new_moves": float(pm[n == 0].mean())}

    def by(col, edges, labels):
        b = x.with_columns(bin=pl.col(col).cut(edges, labels=labels)).group_by("bin").agg(
            visits=pl.len(), gain=c("gain").mean(), maia_ll=c("lm").mean(), seen=c("hit").mean()).sort("bin")
        return {str(r["bin"]): {k: r[k] for k in ("visits", "gain", "maia_ll", "seen")} for r in b.to_dicts()}

    out["by_N_prev"] = by("N_prev", [1, 2, 4, 8, 16], ["1", "2", "3-4", "5-8", "9-16", "17+"])
    out["by_ply"] = by("ply", [0, 9, 19, 39], ["0", "1-9", "10-19", "20-39", "40-79"])
    out["by_position_visits"] = by("pos_total", [99, 999, 9999, 99999], ["<100", "100-999", "1k-10k", "10k-100k",
                                                                          "100k+"])
    # per player: mean gain over their scored visits, and a split-half check that the gain is a stable trait
    per = (x.with_columns(half=(c("g") % 2)).group_by("player", "half").agg(gain=c("gain").mean(), k=pl.len()))
    pw = per.pivot(on="half", index="player", values=["gain", "k"])
    pw = pw.filter((c("k_0") >= 30) & (c("k_1") >= 30))
    from scipy.stats import spearmanr
    allp = x.group_by("player").agg(gain=c("gain").mean(), k=pl.len()).filter(c("k") >= 60)
    q = allp["gain"].quantile
    out["per_player"] = {"players_ge60": allp.height,
                         "gain_quantiles": {str(p): float(q(p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)},
                         "share_gain_positive": float((allp["gain"] > 0).mean()),
                         "split_half_rho": float(spearmanr(pw["gain_0"], pw["gain_1"]).statistic),
                         "split_half_players": pw.height}
    print(json.dumps(out, indent=1), flush=True)
    return out


def save(section: str, obj: dict):
    r = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    r[section] = obj
    RESULTS.write_text(json.dumps(r, indent=1) + "\n")
    print(f"-> {RESULTS.name} {section}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["coverage", "score"])
    a = ap.parse_args()
    d = visits()
    save(a.cmd, coverage(d) if a.cmd == "coverage" else score(d))
