"""Does style transfer across time controls? Same player at 3+0 vs 5+0 (2026-10 held-out, CPU, no engine).

    python -m doppel_lab.style_transfer

Held-out players with at least MIN_EACH games at both 3+0 and 5+0. Per player and time control, the mean of each
persistence.py side feature. Cross-time-control Spearman (3+0 mean vs 5+0 mean) against the within-3+0 split-half
Spearman of the same players (odd vs even 3+0 games): transfer ratio = cross / within. Think time is also
compared as a fraction of the base clock, since 5+0 has 1.67x the time.
Output: lab/results/style_transfer.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from doppel_lab.persistence import CORPUS, FEATS, RESULTS, held_out, side_feats

MIN_EACH = 20
c = pl.col


def main():
    g = pl.concat([pl.read_parquet(f) for f in sorted(CORPUS.glob("part-*.parquet"))])
    g = g.filter(((c("base") == 180) | (c("base") == 300)) & (c("inc") == 0))
    tc = g.select(c("base"), p=pl.concat_list([c("white").str.to_lowercase(), c("black").str.to_lowercase()])).explode("p")
    cnt = tc.group_by("p", "base").len().pivot(on="base", index="p", values="len").fill_null(0)
    keep = cnt.filter((c("180") >= MIN_EACH) & (c("300") >= MIN_EACH))["p"].to_list()
    keep = set(p for p in keep if held_out(p))
    g = g.filter(c("white").str.to_lowercase().is_in(list(keep)) | c("black").str.to_lowercase().is_in(list(keep)))
    print(f"{len(keep):,} players with >= {MIN_EACH} games at both 3+0 and 5+0", flush=True)
    parts = []
    for base in (180, 300):
        gb = g.filter(c("base") == base)
        s = side_feats(gb).filter(c("player").is_in(list(keep))).with_columns(base=pl.lit(base))
        parts.append(s.with_columns(k=pl.int_range(pl.len()).over("player")))
    s = pl.concat(parts, how="diagonal").with_columns(think_frac=c("think_mean") / c("base"))
    feats = FEATS + ["think_frac"]
    by_tc = s.group_by("player", "base").agg(*[c(f).fill_nan(None).mean() for f in feats])
    a = by_tc.filter(c("base") == 180).sort("player")
    b = by_tc.filter(c("base") == 300).sort("player")
    a, b = a.join(b.select("player"), on="player"), b.join(a.select("player"), on="player")
    h = s.filter(c("base") == 180).with_columns(half=c("k") % 2).group_by("player", "half").agg(
        *[c(f).fill_nan(None).mean() for f in feats])
    h0, h1 = (h.filter(c("half") == i).sort("player").join(a.select("player"), on="player") for i in (0, 1))
    res = {"players": a.height, "features": {}}
    for f in feats:
        x, y = a[f].to_numpy().astype(float), b[f].to_numpy().astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        cross = spearmanr(x[ok], y[ok]).statistic
        u, v = h0[f].to_numpy().astype(float), h1[f].to_numpy().astype(float)
        ok2 = np.isfinite(u) & np.isfinite(v)
        within = spearmanr(u[ok2], v[ok2]).statistic
        res["features"][f] = {"cross_tc_rho": float(cross), "within_3p0_split_half_rho": float(within),
                              "transfer_ratio": float(cross / within) if within else None,
                              "mean_3p0": float(np.nanmean(x)), "mean_5p0": float(np.nanmean(y))}
        print(f"{f:14s} cross 3+0 vs 5+0 rho {cross:+.3f}  within 3+0 {within:+.3f}  ratio {cross / within:.2f}  "
              f"(means {np.nanmean(x):.3f} -> {np.nanmean(y):.3f})", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "style_transfer.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
