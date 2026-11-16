"""Prototype of an explicit per-player repertoire memory for the Doppel, scored on held-out repeat visits (CPU).

    python -m doppel_lab.repertoire_memory --band 1500     # needs data/v0e/repeat_v0f[_<band>].parquet
                                                           # (repeat_v0f --dump, one short GPU job)

repeat_v0f (SCIENCE.md §8) showed z holds ~1% of what a player's own earlier moves at a repeated position say, and
that the plain count memory  p = (n_prev(move) + a p_model(move)) / (N_prev + a)  with a = 0.5 gains ~0.9 nats per
repeat visit. This asks what a deployable memory should look like, on the same valrich visits (online: only
earlier games), mixing into v0f's posterior ("post") since that is what the Doppel plays from:

  count      the baseline above, a fitted
  recency    earlier visits weighted lam^(visits since): do players drift, so the last time matters most?
  last       only the most recent visit's move (lam -> 0)
  adaptive   a depends on the player's running consistency (share of their earlier repeat visits that repeated
             their previous move there) and N_prev: a fitted per cell on half the players, scored on the other half
Per variant: gain over post in nats per repeat visit, player-bootstrap CI, and per target game (sum over a side's
repeat visits / sides with any decision positions), next to v0f's own whole-game gain over Maia per side.
Output: lab/results/repertoire_memory.json[band]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results/repertoire_memory.json"
ALPHAS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
LAMS = (1.0, 0.9, 0.8, 0.6, 0.4, 0.2)
c = pl.col


def weighted_hist(x: pl.DataFrame, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Per repeat visit: decayed weight of earlier visits with the same move (w_same) and of all earlier (w_all).
    x is the full visit table (first visits included), sorted player, key, t, g."""
    pk = x.select(grp=pl.struct("player", "key").rank("dense")).to_series().to_numpy()
    mv = x["mv"].to_numpy()
    same, tot = np.zeros(len(x)), np.zeros(len(x))
    starts = np.flatnonzero(np.r_[True, pk[1:] != pk[:-1]])
    ends = np.r_[starts[1:], len(x)]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        m = mv[s:e]
        for i in range(1, e - s):
            w = lam ** np.arange(i - 1, -1, -1.0)       # most recent earlier visit weight 1
            tot[s + i] = w.sum()
            same[s + i] = w[m[:i] == m[i]].sum()
    return same, tot


def mix(same, tot, p, a):
    return np.log((same + a * p) / (tot + a) + 1e-12)


def ci(pid, gain, rng):
    u, inv = np.unique(pid, return_inverse=True)
    s, n = np.bincount(inv, gain), np.bincount(inv)
    b = [s[i].sum() / n[i].sum() for i in (rng.integers(0, len(u), len(u)) for _ in range(1000))]
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


def main(band: int):
    run = "v0f" if band == 1500 else f"v0f_{band}"
    from doppel_lab.repeat_v0f import visits
    from doppel_lab.v0e_data import Split
    sp = Split("valrich", band)
    allv = visits(sp, band).sort("player", "key", "t", "g")
    rep = pl.read_parquet(ROOT / f"data/v0e/repeat_{run}.parquet").select("g", "ply", "maia", "post")
    x = allv.join(rep, on=["g", "ply"], how="left")
    r = x["N_prev"].to_numpy() >= 1
    assert x.filter(c("N_prev") >= 1)["post"].null_count() == 0
    pid = x["player"].to_numpy()[r]
    post, maia = x["post"].to_numpy()[r], x["maia"].to_numpy()[r]
    pp = np.exp(post)
    rng = np.random.default_rng(0)
    test = (np.unique(pid, return_inverse=True)[1] * 2654435761 % 1000) < 500   # player halves
    out = {"band": band, "run": run, "repeat_visits": int(r.sum()), "players": int(len(np.unique(pid))),
           "maia_minus_post": float((maia - post).mean())}

    # count + recency grid (a and lam fitted on the fit half, reported on the test half)
    grid = {}
    for lam in LAMS:
        same, tot = weighted_hist(x, lam)
        same, tot = same[r], tot[r]
        for a in ALPHAS:
            g = mix(same, tot, pp, a) - post
            grid[(lam, a)] = g
    fit = {k: float(v[~test].mean()) for k, v in grid.items()}
    out["grid_fit_half"] = {f"lam{l}_a{a}": v for (l, a), v in fit.items()}
    variants = {}
    best_a1 = max(ALPHAS, key=lambda a: fit[(1.0, a)])
    variants["count"] = (1.0, best_a1)
    variants["recency"] = max(fit, key=fit.get)
    variants["last"] = (0.2, max(ALPHAS, key=lambda a: fit[(0.2, a)]))
    res = {}
    for name, k in variants.items():
        g = grid[k]
        res[name] = {"lam": k[0], "a": k[1], "gain_over_post": float(g[test].mean()),
                     "ci": ci(pid[test], g[test], rng)}

    # adaptive a: cells of (running consistency bin, N_prev bin), a fitted per cell on the fit half
    lam_b = variants["recency"][0]
    same1, tot1 = weighted_hist(x, 1.0)
    rep_prev = (same1 > 0)                                        # did this visit repeat an earlier move?
    xs = x.with_columns(rp=pl.Series(rep_prev.astype(np.float64)), isrep=c("N_prev") >= 1)
    xs = xs.sort("player", "t", "g", "ply").with_columns(           # running consistency over earlier games only
        cs=(c("rp") * c("isrep")).cum_sum().over("player"), cn=c("isrep").cast(pl.Int64).cum_sum().over("player"),
        gfirst=pl.int_range(pl.len()).over("player", "g") == 0)
    # exclude the current game's visits: use the totals as of the start of the game
    xs = xs.with_columns(cs0=pl.when(c("gfirst")).then(c("cs") - c("rp") * c("isrep")).otherwise(None),
                         cn0=pl.when(c("gfirst")).then(c("cn") - c("isrep").cast(pl.Int64)).otherwise(None))
    xs = xs.with_columns(cs0=c("cs0").forward_fill().over("player", "g"), cn0=c("cn0").forward_fill().over("player", "g"))
    xs = xs.sort("player", "key", "t", "g")
    cons = ((xs["cs0"].to_numpy() + 1) / (xs["cn0"].to_numpy() + 2))[r]       # Laplace-smoothed
    cbin = np.digitize(cons, [0.5, 0.65, 0.8, 0.9])
    nbin = np.digitize(x["N_prev"].to_numpy()[r], [2, 4, 8])
    cell = cbin * 4 + nbin
    same_b, tot_b = (lambda s, t: (s[r], t[r]))(*weighted_hist(x, lam_b))
    cand = {a: mix(same_b, tot_b, pp, a) - post for a in ALPHAS}
    ga = np.zeros(r.sum())
    cells = {}
    for k in np.unique(cell):
        m = cell == k
        a = max(ALPHAS, key=lambda a: cand[a][m & ~test].mean() if (m & ~test).any() else -1e9)
        ga[m] = cand[a][m]
        cells[f"c{k // 4}_n{k % 4}"] = {"a": a, "visits": int(m.sum()), "consistency_mean": float(cons[m].mean()),
                                        "gain_test": float(cand[a][m & test].mean()) if (m & test).any() else None}
    res["adaptive"] = {"lam": lam_b, "gain_over_post": float(ga[test].mean()), "ci": ci(pid[test], ga[test], rng),
                       "cells": cells}
    out["variants"] = res

    # per-game accounting on the 600 target players (48 games each; their opponents mostly have 1-2): repeat
    # visits per target side and share of their own decision plies, times the best variant's gain there
    best = max(res, key=lambda k: res[k]["gain_over_post"])
    out["best_variant"] = best
    tp = set(pl.read_parquet(ROOT / f"data/v0e/valrich_players_{band}.parquet")["player"].to_list())
    tid = np.array([i for i, n in enumerate(np.asarray(sp.players)) if n in tp])
    istgt = np.isin(sp.pid, tid)
    plies = np.diff(sp.ix.off)
    own = sum(((plies + 1 - col) // 2)[istgt[:, col]].sum() for col in (0, 1))
    tv = np.isin(pid, tid)
    gbest = {"count": grid[variants["count"]], "recency": grid[variants["recency"]], "last": grid[variants["last"]],
             "adaptive": ga}[best]
    out["target"] = {"players": len(tid), "sides": int(istgt.sum()), "own_plies_per_side": float(own / istgt.sum()),
                     "repeat_visits_per_side": float(tv.sum() / istgt.sum()), "repeat_share_of_own_plies": float(tv.sum() / own),
                     "gain_per_repeat_visit": float(gbest[tv].mean()),
                     "gain_per_side_nats": float(gbest[tv].sum() / istgt.sum()),
                     "gain_per_own_ply_nats": float(gbest[tv].sum() / own),
                     "v0f_gain_per_repeat_visit_over_maia": float((post - maia)[tv].mean())}
    allres = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    allres[str(band)] = out
    RESULTS.write_text(json.dumps(allres, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "grid_fit_half"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=int, default=1500)
    main(ap.parse_args().band)
