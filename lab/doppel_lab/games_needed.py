"""How many games pin down a player? Rating from surface style averaged over k games (CPU, no engine).

    python -m doppel_lab.games_needed

Players and per-game side features as persistence.py (2026-10 held-out, 3+0, >= 40 games). For k in KS, each
player's features are averaged over k of their games (the first k in stream order), and a gradient-boosted
regressor predicts the player's mean rating over those games; 5-fold CV grouped by player. Rating is never an
input. Reported: MAE and R² by k, the data-needs curve a Doppel's z has to beat.
Output: lab/results/games_needed.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from doppel_lab.persistence import CORPUS, FEATS, MIN_GAMES, RESULTS, SEED, held_out, side_feats

KS = (1, 2, 5, 10, 20, 40)
N_PLAYERS = 8_000
c = pl.col


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    rng = np.random.default_rng(SEED)
    g = pl.concat([pl.read_parquet(f) for f in sorted(CORPUS.glob("part-*.parquet"))]).with_row_index("t")
    g = g.filter((c("base") == 180) & (c("inc") == 0))
    names = pl.concat([g["white"], g["black"]]).str.to_lowercase().value_counts().filter(c("count") >= max(KS))
    names = names.filter(pl.Series([held_out(n) for n in names["white"].to_list()]))
    keep = sorted(names["white"].to_list())
    keep = set(rng.choice(keep, min(N_PLAYERS, len(keep)), replace=False).tolist())
    g = g.filter(c("white").str.to_lowercase().is_in(list(keep)) | c("black").str.to_lowercase().is_in(list(keep)))
    s = side_feats(g).filter(c("player").is_in(list(keep)))
    s = s.with_columns(k=pl.int_range(pl.len()).over("player"))
    print(f"{len(keep):,} players with >= {max(KS)} 3+0 games", flush=True)
    res = {"players": len(keep), "ks": {}}
    for k in KS:
        a = s.filter(c("k") < k).group_by("player").agg(c("elo").mean(), *[c(f).fill_nan(None).mean() for f in FEATS])
        X, y = a.select(FEATS).to_numpy().astype(float), a["elo"].to_numpy().astype(float)
        use = [j for j in range(X.shape[1]) if len(np.unique(X[np.isfinite(X[:, j]), j])) >= 2]   # HGB needs 2+ values
        X = X[:, use]
        fold = (a["player"].hash(SEED) % 5).to_numpy()
        pred = np.empty(len(y))
        for f in range(5):
            tr, te = fold != f, fold == f
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        r = {"mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred)),
             "mae_const": float(np.abs(y - np.median(y)).mean())}
        res["ks"][str(k)] = r
        print(f"k={k:3d} games: rating MAE {r['mae']:.0f} (constant {r['mae_const']:.0f}), R² {r['r2']:.3f}", flush=True)
    (RESULTS / "games_needed.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
