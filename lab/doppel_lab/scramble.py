"""Time scrambles: does rating matter less when both clocks run down? (2026-10 corpus, blitz, CPU)

    python -m doppel_lab.scramble

Per game, the lowest clock each side reached (from per-ply [%clk]). Scramble = both sides under SCR seconds at
some point; calm = neither under 30 s. For each gap bin, the higher-rated side's score in scrambles vs calm games,
the Elo expectation, and the fitted Elo scale s (E = 1 / (1 + 10^(-gap/s))): a larger s means rating matters less.
Excludes 1500-1599 ratings (provisional starts, see elo_curve.py).
Output: lab/results/scramble.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "2026-10"
RESULTS = ROOT / "lab" / "results"
SCR = 10
c = pl.col


def main():
    d = pl.scan_parquet(str(CORPUS / "*" / "*.parquet")).select("white_elo", "black_elo", "result", "termination",
                                                                 "clocks", "base").collect()
    d = d.filter(~c("white_elo").is_between(1500, 1599) & ~c("black_elo").is_between(1500, 1599)
                 & (c("clocks").list.len() >= 20))
    d = d.with_columns(wmin=c("clocks").list.gather_every(2).list.min(),
                       bmin=c("clocks").list.gather_every(2, offset=1).list.min()).drop("clocks")
    d = d.with_columns(state=pl.when((c("wmin") < SCR) & (c("bmin") < SCR)).then(pl.lit("scramble"))
                       .when((c("wmin") >= 30) & (c("bmin") >= 30)).then(pl.lit("calm")).otherwise(pl.lit("one-sided")),
                       sw=pl.when(c("result") == "1-0").then(1.0).when(c("result") == "0-1").then(0.0).otherwise(0.5),
                       g=(c("white_elo") - c("black_elo")).cast(pl.Float64))
    d = d.with_columns(gap=c("g").abs(), s=pl.when(c("g") >= 0).then(c("sw")).otherwise(1 - c("sw")))
    d = d.filter(c("gap").is_between(1, 400)).with_columns(gb=(c("gap") // 50 * 50 + 25))
    res = {"scr_seconds": SCR, "states": {}}
    print(d.group_by("state").agg(n=pl.len(), draw=(c("result") == "1/2-1/2").mean(),
                                  flag=(c("termination") == "Time forfeit").mean()).sort("state"))
    for st in ("calm", "one-sided", "scramble"):
        b = d.filter(c("state") == st).group_by("gb").agg(score=c("s").mean(), n=pl.len()).sort("gb")
        gap, sc, n = (b[k].to_numpy().astype(float) for k in ("gb", "score", "n"))
        f = lambda s_: float((n * (sc - 1 / (1 + 10 ** (-gap / s_))) ** 2).sum())
        s_fit = minimize_scalar(f, bounds=(150, 5000), method="bounded").x
        res["states"][st] = {"games": int(n.sum()), "scale": float(s_fit),
                             "by_gap": [{"gap": float(a), "score": float(x), "n": int(m)} for a, x, m in zip(gap, sc, n)]}
        pick = {int(a): round(float(x), 3) for a, x in zip(gap, sc) if a in (75, 175, 275, 375)}
        print(f"{st:9s} games {int(n.sum()):>10,}  fitted scale {s_fit:6.0f}  favourite's score by gap {pick}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "scramble.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
