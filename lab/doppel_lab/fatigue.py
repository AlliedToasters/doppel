"""Session fatigue: do players do worse, or play faster, the longer a session runs? (Lichess heads 2018-2026, CPU)

    python -m doppel_lab.fatigue

Sessions and sides as tilt.py (games chained by start gaps <= 60 min). Regressors: dummies for the game's
index in its session (2, 3-4, 5-8, 9-16, 17+; game 1 is the base), with session fixed effects, so a session is
compared with itself. The regressor is position, not a lagged outcome, so there's no finite-sequence bias,
and the Elo expectation uses each game's own pre-game ratings, so rating drift during a session is absorbed.
Outcomes: score minus expectation, mean think per move, flag loss, early loss.
Stopping bias: players stop more often after a loss (tilt.py), so a session's final game skews toward losses and
is never game 1. The "no_final" estimate drops every session's final game (sessions of 3+ games).
Output: lab/results/fatigue.json
"""

from __future__ import annotations

import json

import polars as pl

from doppel_lab.tilt import GAP_MIN, LAKE, RESULTS, sides_of, within

BINS = [(2, 2, "g2"), (3, 4, "g3_4"), (5, 8, "g5_8"), (9, 16, "g9_16"), (17, 10_000, "g17p")]
c = pl.col


def main():
    s = pl.concat([sides_of(p) for p in sorted(LAKE.glob("????-??")) if int(p.name[:4]) >= 2018])
    gap = lambda a, b: (a - b).dt.total_seconds() / 60
    s = s.sort("player", "t").with_columns(prev_t=c("t").shift(1).over("player"))
    s = s.with_columns(new=c("prev_t").is_null() | (gap(c("t"), c("prev_t")) > GAP_MIN))
    s = s.with_columns(session=c("new").cum_sum()).with_columns(idx=pl.int_range(1, pl.len() + 1).over("session"),
                                                                n_sess=pl.len().over("session"))
    s = s.filter(c("n_sess") >= 2)
    s = s.with_columns([c("idx").is_between(lo, hi).cast(pl.Float64).alias(n) for lo, hi, n in BINS])
    xs = [n for *_, n in BINS]
    print(f"{s.height:,} sides in {s['session'].n_unique():,} sessions of 2+ games", flush=True)
    res = {"sides": s.height, "outcomes": {}, "no_final": {}}
    nf = s.filter((c("n_sess") >= 3) & (c("idx") < c("n_sess")))
    for key, d in (("outcomes", s), ("no_final", nf)):
        print(f"-- {key}: {d.height:,} sides", flush=True)
        for y in ["resid", "think_mean", "flag_loss", "early_loss"]:
            r = within(d.with_columns(player=c("session")), y, xs)
            res[key][y] = r
            print(f"{y:11s} vs game 1: " + "  ".join(f"{x} {r[x]['coef']:+.4f} (z {r[x]['z']:+.1f})" for x in xs), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "fatigue.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
