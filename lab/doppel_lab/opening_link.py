"""Does a player's opening play predict their middlegame style? (CPU, ~5 min.)

    python -m doppel_lab.opening_link        # -> lab/results/opening_link.json

The Allie / Maia-3 benchmark scores plies >= 10 only, but plies 0-9 (the test game's own and every earlier game's)
are legitimate context. If opening behaviour predicts middlegame traits, the opening is a cheap z input; if not,
it is only repertoire. Data: the 1500-band train table with net evals (moves_f1_T2net_1500_train, 700k games,
26.8k players with >= 16 games), the same player split as sharpness.style (odd / even games).

  middlegame traits   sharpness.side_table at plies >= 10: steering, allocation, Maia log-prob, entropy faced,
                      obvious misses, blunder rate, ... (sharpness.TRAITS)
  opening features    own moves at plies 0-9: Maia log-prob, played rank > 1 share, rank > 3 share, entropy
                      faced, think seconds, think fraction, net win-chance loss; first-move shares (White's
                      1.e4 / 1.d4 / 1.c4 / 1.Nf3, Black's ...c5 / e5 / e6 / c6 / d5 / Nf6 / g6)
Per trait: ridge from half-A opening features to the half-B trait (all rating-residualised, 5-fold CV over
players), scored as Spearman of prediction vs truth. References: the trait's own split-half rho (the ceiling a
half-A predictor can reach) and rating alone (0 by construction after residualising). Reported as rho and as a
share of the ceiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from doppel_lab import sharpness as sh

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "data/v0/features/moves_f1_T2net_1500_train"
RESULTS = ROOT / "lab/results/opening_link.json"
WHITE1 = ("e2e4", "d2d4", "c2c4", "g1f3")
BLACK1 = ("c7c5", "e7e5", "e7e6", "c7c6", "d7d5", "g8f6", "g7g6")
c = pl.col


def opening_table() -> pl.DataFrame:
    t = (pl.scan_parquet(TABLE / "part-*.parquet")
         .select("game_id", "mover", "ply", "uci", "maia_played_lp", "maia_played_rank", "maia_entropy", "think_s",
                 "think_frac", "wc_loss")
         .filter(c("ply") < 10).collect())
    first = t.filter(c("ply") < 2).select("game_id", "mover", f1="uci")
    o = t.group_by("game_id", "mover").agg(
        o_lp=c("maia_played_lp").mean(), o_rank1=(c("maia_played_rank") > 1).mean(),
        o_rank3=(c("maia_played_rank") > 3).mean(), o_ent=c("maia_entropy").mean(), o_think=c("think_s").mean(),
        o_tfrac=c("think_frac").mean(), o_loss=c("wc_loss").mean())
    o = o.join(first, on=["game_id", "mover"], how="left")
    o = o.with_columns(*[((c("mover") == 0) & (c("f1") == u)).cast(pl.Float64).alias(f"w_{u}") for u in WHITE1],
                       *[((c("mover") == 1) & (c("f1") == u)).cast(pl.Float64).alias(f"b_{u}") for u in BLACK1],
                       is_white=(c("mover") == 0).cast(pl.Float64))
    return o.drop("f1")


def main():
    sh.TABLE = TABLE
    mid = sh.side_table()
    op = opening_table()
    OPEN = [k for k in op.columns if k not in ("game_id", "mover")]
    g = pl.read_parquet(sh.GAMES, columns=["game_id", "white", "black", "white_elo", "black_elo"])
    s = mid.join(op, on=["game_id", "mover"]).join(g, on="game_id").with_columns(
        player=pl.when(c("mover") == 0).then(c("white")).otherwise(c("black")).str.to_lowercase(),
        elo=pl.when(c("mover") == 0).then(c("white_elo")).otherwise(c("black_elo")).cast(pl.Float64))
    keep = s.group_by("player").len().filter(c("len") >= sh.MIN_GAMES)["player"]
    p = (s.filter(c("player").is_in(keep.implode())).sort("player", "game_id")
         .with_columns(half=pl.int_range(pl.len()).over("player") % 2))
    agg = p.group_by("player", "half").agg(c("elo").mean(), *[c(k).mean() for k in sh.TRAITS + OPEN])
    A = agg.filter(c("half") == 0).sort("player")
    B = agg.filter(c("half") == 1).sort("player")
    A, B = A.join(B.select("player"), on="player"), B.join(A.select("player"), on="player")
    elo = ((A["elo"] + B["elo"]) / 2).to_numpy()
    print(f"{A.height:,} players", flush=True)

    def resid(x):
        x = np.asarray(x, float)
        ok = np.isfinite(x)
        Z = np.c_[np.ones(ok.sum()), elo[ok], elo[ok] ** 2]
        r = np.full_like(x, np.nan)
        r[ok] = x[ok] - Z @ np.linalg.lstsq(Z, x[ok], rcond=None)[0]
        return r

    def std(M):
        return (M - np.nanmean(M, 0)) / np.maximum(np.nanstd(M, 0), 1e-9)

    XA = np.column_stack([resid(A[k].to_numpy()) for k in OPEN])
    XB = np.column_stack([resid(B[k].to_numpy()) for k in OPEN])
    usable = (np.isfinite(XA).mean(0) > 0.5) & (np.isfinite(XB).mean(0) > 0.5)
    dropped = [k for k, u in zip(OPEN, usable) if not u]
    print("dropped (mostly missing):", dropped, flush=True)
    OPEN = [k for k, u in zip(OPEN, usable) if u]
    XA, XB = np.nan_to_num(std(XA[:, usable])), np.nan_to_num(std(XB[:, usable]))   # missing -> the mean
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, A.height)

    def cv_rho(X, y, lam=10.0):
        ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
        pred = np.full(len(y), np.nan)
        for f in range(5):
            tr, te = ok & (fold != f), ok & (fold == f)
            Xt = np.c_[np.ones(tr.sum()), X[tr]]
            w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T @ y[tr])
            pred[te] = np.c_[np.ones(te.sum()), X[te]] @ w
        return float(spearmanr(pred[ok], y[ok]).statistic), pred

    out = {"players": A.height, "opening_features": OPEN, "dropped": dropped, "traits": {}}
    for k in sh.TRAITS:
        ya, yb = resid(A[k].to_numpy()), resid(B[k].to_numpy())
        ok = np.isfinite(ya) & np.isfinite(yb)
        ceil = float(spearmanr(ya[ok], yb[ok]).statistic)
        cross = (cv_rho(XA, yb)[0] + cv_rho(XB, ya)[0]) / 2        # opening in one half -> trait in the other
        same = (cv_rho(XA, ya)[0] + cv_rho(XB, yb)[0]) / 2         # same half (shares game-level noise)
        # strongest single opening feature, cross-half
        single = {f: float((spearmanr(XA[ok, i], yb[ok]).statistic + spearmanr(XB[ok, i], ya[ok]).statistic) / 2)
                  for i, f in enumerate(OPEN)}
        top = sorted(single.items(), key=lambda kv: -abs(kv[1]))[:3]
        out["traits"][k] = {"split_half_rho": ceil, "opening_cross_half_rho": cross, "share_of_ceiling":
                            cross / ceil if ceil > 0 else None, "opening_same_half_rho": same,
                            "top_single_features": dict(top)}
        print(f"{k:13s} ceiling {ceil:+.3f}  opening->other half {cross:+.3f} ({cross / ceil if ceil > 0 else 0:.0%})"
              f"  same half {same:+.3f}  top {', '.join(f'{f} {v:+.2f}' for f, v in top)}", flush=True)
    RESULTS.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
