"""Do players adapt to the opponent? Off-repertoire openings and pace vs the rating gap, within player (CPU).

    python -m doppel_lab.opp_adapt

2026-10 held-out corpus, 3+0. Per (player, colour) with >= 30 games: the modal first move as White, the modal
reply to 1.e4 / 1.d4 as Black. Per game: off = the player didn't play their modal move. Regressors: opponent
rating minus own, in bins (vs |gap| < 50): -300 or less, -300..-150, -150..-50, +50..+150, +150..+300, +300 or
more. Player x colour fixed effects (tilt.within). Also mean think per own move (the same clocks rule).
Output: lab/results/opp_adapt.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from doppel_lab.persistence import CORPUS, RESULTS, held_out
from doppel_lab.tilt import within

BINS = [(-9999, -300, "opp_m300"), (-300, -150, "opp_m150"), (-150, -50, "opp_m50"), (50, 150, "opp_p50"),
        (150, 300, "opp_p150"), (300, 9999, "opp_p300")]
c = pl.col


def main():
    g = pl.concat([pl.read_parquet(f, columns=["white", "black", "white_elo", "black_elo", "moves", "clocks", "base", "inc"])
                   for f in sorted(CORPUS.glob("part-*.parquet"))])
    g = g.filter((c("base") == 180) & (c("inc") == 0) & (c("moves").list.len() >= 12))
    w = g.select(p=c("white").str.to_lowercase(), gap=c("black_elo") - c("white_elo"), mv=c("moves").list.get(0),
                 clocks=c("clocks").list.gather_every(2), col=pl.lit("w"))
    b = g.filter(c("moves").list.get(0).is_in(["e2e4", "d2d4"])).select(
        p=c("black").str.to_lowercase(), gap=c("white_elo") - c("black_elo"),
        mv=c("moves").list.get(0) + c("moves").list.get(1), clocks=c("clocks").list.gather_every(2, offset=1),
        col=pl.lit("b"))
    d = pl.concat([w, b]).with_columns(key=c("p") + ":" + c("col"))
    d = d.with_columns(n=pl.len().over("key")).filter(c("n") >= 30)
    d = d.filter(pl.Series([held_out(x) for x in d["p"].to_list()]))
    mode = d.group_by("key", "mv").len().sort("len", descending=True).unique("key", keep="first").select("key", modal="mv")
    d = d.join(mode, on="key").with_columns(off=(c("mv") != c("modal")).cast(pl.Float64))
    th = [float(np.clip(np.asarray(a[:-1], float) - np.asarray(a[1:], float), 0, 300)[1:].mean()) if len(a) > 4 else None
          for a in d["clocks"].to_list()]
    d = d.with_columns(think=pl.Series(th, dtype=pl.Float64)).drop("clocks")
    d = d.with_columns([c("gap").is_between(lo, hi, closed="left").cast(pl.Float64).alias(n) for lo, hi, n in BINS])
    xs = [n for *_, n in BINS]
    print(f"{d.height:,} sides, {d['key'].n_unique():,} player-colours; off-repertoire share {d['off'].mean():.3f}", flush=True)
    res = {"sides": d.height, "off_share": float(d["off"].mean()), "outcomes": {}}
    for y in ["off", "think"]:
        r = within(d.with_columns(player=c("key")), y, xs)
        res["outcomes"][y] = r
        print(f"{y:6s} vs |gap| < 50: " + "  ".join(f"{x} {r[x]['coef']:+.4f} (z {r[x]['z']:+.1f})" for x in xs), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "opp_adapt.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
