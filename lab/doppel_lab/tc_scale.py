"""Does rating mean more at slower time controls? Fitted Elo scale per time control (2026-10 corpus, CPU).

    python -m doppel_lab.tc_scale

As scramble.py: favourite's score by 50-point gap bin (gap 1-400, 1500-1599 ratings excluded), fitted scale s in
E = 1 / (1 + 10^(-gap/s)); s = 400 is nominal Elo, larger means ratings predict less. Per time control with
>= 200k games, plus draw and flag rates.
Output: lab/results/tc_scale.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

from doppel_lab.scramble import CORPUS, RESULTS

c = pl.col


def main():
    d = pl.scan_parquet(str(CORPUS / "*" / "*.parquet")).select("white_elo", "black_elo", "result", "termination",
                                                                 "base", "inc").collect()
    d = d.filter(~c("white_elo").is_between(1500, 1599) & ~c("black_elo").is_between(1500, 1599))
    d = d.with_columns(sw=pl.when(c("result") == "1-0").then(1.0).when(c("result") == "0-1").then(0.0).otherwise(0.5),
                       g=(c("white_elo") - c("black_elo")).cast(pl.Float64),
                       tc=c("base").cast(pl.Int64).cast(pl.String) + "+" + c("inc").cast(pl.Int64).cast(pl.String))
    d = d.with_columns(gap=c("g").abs(), s=pl.when(c("g") >= 0).then(c("sw")).otherwise(1 - c("sw")))
    big = d.group_by("tc").len().filter(c("len") >= 200_000)["tc"].to_list()
    d = d.filter(c("gap").is_between(1, 400) & c("tc").is_in(big)).with_columns(gb=c("gap") // 50 * 50 + 25)
    res = {}
    for tc in sorted(big, key=lambda t: (int(t.split("+")[0]) + 40 * int(t.split("+")[1]))):
        x = d.filter(c("tc") == tc)
        b = x.group_by("gb").agg(score=c("s").mean(), n=pl.len()).sort("gb")
        gap, sc, n = (b[k].to_numpy().astype(float) for k in ("gb", "score", "n"))
        f = lambda s_: float((n * (sc - 1 / (1 + 10 ** (-gap / s_))) ** 2).sum())
        s_fit = minimize_scalar(f, bounds=(150, 5000), method="bounded").x
        res[tc] = {"games": x.height, "scale": float(s_fit), "draw": float((x["result"] == "1/2-1/2").mean()),
                   "flag": float((x["termination"] == "Time forfeit").mean())}
        print(f"{tc:7s} games {x.height:>10,}  scale {s_fit:5.0f}  draws {res[tc]['draw']:.3f}  flags {res[tc]['flag']:.3f}",
              flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tc_scale.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
