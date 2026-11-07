"""Pooling z over a player's games: does averaging the v0d z (or Maia-3's embedding) over k games beat surface style?

    python -m doppel_lab.players_suite build                 # CPU: data/bench/players/games.parquet + sides
    # GPU (warden): python -m doppel_lab.bench_embed run --suite players --gpu-ok
    python -m doppel_lab.players_suite score                 # CPU: v0d z on CPU, then pooled probes

Held-out 2026-10 players with >= MIN_GAMES 3+0 games; N_PLAYERS of them; their first PER games (time order).
A cheap preview of a multi-game z with no training: each game gives a per-side vector (v0d posterior mean
over both halves, 8 dims; Maia-3 mean final-layer embedding, PCA 64; surface style as persistence.py), and a
player's vector is the mean over their first k games. Scored as games_needed.py (rating R², trees, 5-fold by
player) and reid_k.py (first k games vs a gallery of games PER-40..PER, pools of 500).
Output: lab/results/players_pool.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl

from doppel_lab.bench import BENCH
from doppel_lab.persistence import CORPUS, FEATS, RESULTS, SEED, held_out, side_feats

D = BENCH / "players"
MIN_GAMES, N_PLAYERS, PER, GALLERY, POOL = 100, 2_000, 80, 40, 500
KS = (1, 2, 5, 10, 20, 40)
c = pl.col


def build():
    rng = np.random.default_rng(SEED)
    g = pl.concat([pl.read_parquet(f) for f in sorted(CORPUS.glob("part-*.parquet"))])
    g = g.filter((c("base") == 180) & (c("inc") == 0) & (c("moves").list.len() >= 20)).with_row_index("t")
    names = pl.concat([g["white"], g["black"]]).str.to_lowercase().value_counts().filter(c("count") >= MIN_GAMES)
    names = [n for n in names["white"].to_list() if held_out(n)]
    keep = sorted(rng.choice(sorted(names), min(N_PLAYERS, len(names)), replace=False).tolist())
    sides = pl.concat([g.select("t", "game_id", player=c(col).str.to_lowercase(), col=pl.lit(i, pl.Int8))
                       for i, col in enumerate(("white", "black"))]).filter(c("player").is_in(keep)).sort("t")
    sides = sides.with_columns(k=pl.int_range(pl.len()).over("player")).filter(c("k") < PER)
    games = g.filter(c("game_id").is_in(sides["game_id"].unique().implode()))
    D.mkdir(parents=True, exist_ok=True)
    games.select("game_id", "moves", "clocks", "base", "inc").write_parquet(D / "games.parquet")
    sides.write_parquet(D / "sides.parquet")
    sf = side_feats(games.with_columns(t=c("t")))
    sf = sf.join(sides.select("t", "player", "k"), on=["t", "player"])
    sf.write_parquet(D / "surface_sides.parquet")
    print(f"{len(keep):,} players, {sides.height:,} sides in {games.height:,} games -> {D}", flush=True)


def score(run: str = "v0d"):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score

    from doppel_lab.bench_doppel import embed
    sides = pl.read_parquet(D / "sides.parquet")
    gids, mu = embed(run, "players")                     # (G, colour, half, 8)
    z = np.load(D / "maia3.npz", allow_pickle=True)
    assert (z["game_ids"] == gids).all()
    gi = {g: i for i, g in enumerate(gids)}
    idx = np.array([gi[g] for g in sides["game_id"].to_list()])
    col = sides["col"].to_numpy()
    zd = np.nanmean(np.where(mu[idx, col] == 0, np.nan, mu[idx, col]), axis=1)          # mean of the two halves
    emb = z["emb"][idx, col].astype(np.float32)
    rng = np.random.default_rng(SEED)
    pca = PCA(64, random_state=SEED).fit(emb[rng.choice(len(emb), min(50_000, len(emb)), replace=False)])
    em = pca.transform(emb)
    surf = pl.read_parquet(D / "surface_sides.parquet")
    sides = sides.join(surf.select("t", "player", "elo", *FEATS), on=["t", "player"], how="left")
    S = sides.select(FEATS).to_numpy().astype(float)
    views = {"surface": S, "doppel_z": zd, "maia_emb": em, "surface+doppel_z": np.c_[S, zd], "surface+maia": np.c_[S, em]}
    players = sides["player"].to_numpy()
    k_ = sides["k"].to_numpy()
    elo = sides["elo"].to_numpy().astype(float)
    up = np.unique(players)
    pi = {p: i for i, p in enumerate(up)}
    pidx = np.array([pi[p] for p in players])
    fold = np.array([hash(p) % 5 for p in up])

    def pool(X, mask):
        out = np.zeros((len(up), X.shape[1]))
        cnt = np.zeros((len(up), X.shape[1]))
        ok = np.isfinite(X) & mask[:, None]
        np.add.at(out, pidx, np.where(ok, X, 0))
        np.add.at(cnt, pidx, ok)
        return np.where(cnt > 0, out / np.maximum(cnt, 1), np.nan)

    res = {"players": len(up), "rating_r2": {}, "reid_top1": {}}
    gal_mask = k_ >= PER - GALLERY
    for name, X in views.items():
        G = pool(X, gal_mask)
        mu_, sd_ = np.nanmean(G, 0), np.nanstd(G, 0) + 1e-9
        Gs = np.nan_to_num((G - mu_) / sd_)
        res["rating_r2"][name], res["reid_top1"][name] = {}, {}
        for k in KS:
            m = k_ < k
            P = pool(X, m)
            Ps = np.nan_to_num((P - mu_) / sd_)
            y = pool(elo[:, None], m)[:, 0]
            P = P[:, [j for j in range(P.shape[1]) if len(np.unique(P[np.isfinite(P[:, j]), j])) >= 2]]  # HGB needs 2+ values
            pred = np.empty(len(y))
            for f in range(5):
                tr, te = fold != f, fold == f
                pred[te] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                         random_state=SEED).fit(P[tr], y[tr]).predict(P[te])
            r2 = float(r2_score(y, pred))
            t1 = n = 0
            for _ in range(max(1, len(up) // POOL)):
                ii = rng.choice(len(up), POOL, replace=False)
                Dm = ((Ps[ii][:, None, :] - Gs[ii][None, :, :]) ** 2).sum(-1)
                t1 += (Dm.argmin(1) == np.arange(POOL)).sum(); n += POOL
            res["rating_r2"][name][str(k)], res["reid_top1"][name][str(k)] = r2, t1 / n
        print(f"{name:17s} rating R² " + " ".join(f"k{k}:{res['rating_r2'][name][str(k)]:.3f}" for k in KS)
              + " | re-ID top-1 " + " ".join(f"k{k}:{res['reid_top1'][name][str(k)]:.3f}" for k in KS), flush=True)
    (RESULTS / f"players_pool_{run}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]]()
