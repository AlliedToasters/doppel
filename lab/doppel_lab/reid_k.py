"""Re-identification vs games of evidence: how many games does surface style need to pick a player out? (CPU)

    python -m doppel_lab.reid_k

Players as persistence.py (2026-10 held-out, 3+0, >= 80 games, up to 8,000). Gallery: the mean features of each
player's last 40 games. Probe: the mean of their first k games (disjoint from the gallery). Standardised,
rating-residualised features (persistence.py's style set), nearest neighbour in pools of 1,000 players; top-1
and top-10 by k. Sizes the k a multi-game z encoder needs to see.
Output: lab/results/reid_k.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from doppel_lab.persistence import CORPUS, FEATS, RESULTS, SEED, held_out, side_feats

KS = (1, 2, 5, 10, 20, 40)
GALLERY, MIN_GAMES, N_PLAYERS, POOL = 40, 80, 8_000, 1_000
c = pl.col


def main():
    rng = np.random.default_rng(SEED)
    g = pl.concat([pl.read_parquet(f) for f in sorted(CORPUS.glob("part-*.parquet"))])
    g = g.filter((c("base") == 180) & (c("inc") == 0)).with_row_index("t")
    names = pl.concat([g["white"], g["black"]]).str.to_lowercase().value_counts().filter(c("count") >= MIN_GAMES)
    names = [n for n in names["white"].to_list() if held_out(n)]
    keep = set(rng.choice(sorted(names), min(N_PLAYERS, len(names)), replace=False).tolist())
    g = g.filter(c("white").str.to_lowercase().is_in(list(keep)) | c("black").str.to_lowercase().is_in(list(keep)))
    s = side_feats(g).filter(c("player").is_in(list(keep))).sort("t")   # side_feats stacks White rows before Black
    s = s.with_columns(k=pl.int_range(pl.len()).over("player"), n=pl.len().over("player"))
    gal = s.filter(c("k") >= c("n") - GALLERY).group_by("player").agg(c("elo").mean(), *[c(f).fill_nan(None).mean() for f in FEATS]).sort("player")
    players = gal["player"]
    print(f"{gal.height:,} players with >= {MIN_GAMES} 3+0 games", flush=True)
    G = gal.select(FEATS).to_numpy().astype(float)
    elo_g = gal["elo"].to_numpy().astype(float)
    mu, sd = np.nanmean(G, 0), np.nanstd(G, 0)

    def resid(X, elo):
        out = np.empty_like(X)
        Z = np.c_[np.ones(len(elo)), elo, elo ** 2]
        for j in range(X.shape[1]):
            x = X[:, j]
            ok = np.isfinite(x)
            b = np.linalg.lstsq(Z[ok], x[ok], rcond=None)[0] if ok.sum() > 10 else np.zeros(3)
            out[:, j] = np.where(ok, (x - Z @ b) / sd[j], 0.0)
        return out

    Gr = resid(G, elo_g)
    res = {"players": gal.height, "gallery_games": GALLERY, "pool": POOL, "ks": {}}
    for k in KS:
        pr = s.filter(c("k") < k).group_by("player").agg(c("elo").mean(), *[c(f).fill_nan(None).mean() for f in FEATS])
        pr = pl.DataFrame({"player": players}).join(pr, on="player", how="left")
        P = pr.select(FEATS).to_numpy().astype(float)
        Pr = resid(P, pr["elo"].to_numpy().astype(float))
        t1 = t10 = n = 0
        for _ in range(max(1, len(players) // POOL)):
            idx = rng.choice(len(players), POOL, replace=False)
            D = ((Pr[idx][:, None, :] - Gr[idx][None, :, :]) ** 2).sum(-1)
            rank = (D < D[np.arange(POOL), np.arange(POOL)][:, None]).sum(1)
            t1 += (rank == 0).sum(); t10 += (rank < 10).sum(); n += POOL
        res["ks"][str(k)] = {"top1": t1 / n, "top10": t10 / n}
        print(f"k={k:3d} probe games vs a {GALLERY}-game gallery: top-1 {t1 / n:.3f}  top-10 {t10 / n:.3f} "
              f"(chance {1 / POOL:.3f})", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "reid_k.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
