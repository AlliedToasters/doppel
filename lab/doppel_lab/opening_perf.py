"""White's edge by rating, and which openings over- or under-perform the Elo expectation (2026-10 corpus, CPU).

    python -m doppel_lab.opening_perf

White score minus the Elo expectation from the two ratings, by mean-rating band (1500-1599 sides excluded:
provisional starts). Per 4-ply line (UCI) with >= MIN_GAMES games in a band: the same residual, its SE, and
z; listed: the largest positive and negative residuals per band group. The residual is White's view; an
opening's Black performance is minus that.
Output: lab/results/opening_perf.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "2026-10"
RESULTS = ROOT / "lab" / "results"
MIN_GAMES = 20_000
GROUPS = [(600, 1100, "<1100"), (1100, 1500, "1100-1499"), (1600, 2000, "1600-1999"), (2000, 2900, "2000+")]
c = pl.col


def main():
    d = pl.scan_parquet(str(CORPUS / "*" / "*.parquet")).select("white_elo", "black_elo", "result", "moves").collect()
    d = d.filter(~c("white_elo").is_between(1500, 1599) & ~c("black_elo").is_between(1500, 1599)
                 & (c("moves").list.len() >= 4))
    d = d.with_columns(sw=pl.when(c("result") == "1-0").then(1.0).when(c("result") == "0-1").then(0.0).otherwise(0.5),
                       exp=1 / (1 + 10 ** ((c("black_elo") - c("white_elo")) / 400)),
                       elo=(c("white_elo") + c("black_elo")) / 2,
                       line=c("moves").list.head(4).list.join(" ")).drop("moves")
    d = d.with_columns(r=c("sw") - c("exp"), band=(c("elo") // 200 * 200).clip(600, 2400))
    edge = d.group_by("band").agg(n=pl.len(), white_resid=c("r").mean(), white_score=c("sw").mean(),
                                  draw=(c("result") == "1/2-1/2").mean()).sort("band")
    print(edge)
    res = {"white_edge_by_band": edge.to_dicts(), "lines": {}}
    for lo, hi, name in GROUPS:
        x = d.filter(c("elo").is_between(lo, hi - 1e-9))
        base = float(x["r"].mean())
        t = x.group_by("line").agg(n=pl.len(), r=c("r").mean(), sd=c("r").std()).filter(c("n") >= MIN_GAMES)
        t = t.with_columns(rel=c("r") - base, z=(c("r") - base) / (c("sd") / c("n").sqrt())).sort("rel")
        res["lines"][name] = {"band_mean_resid": base, "worst": t.head(6).to_dicts(), "best": t.tail(6).reverse().to_dicts()}
        print(f"\n{name}: White resid {base:+.4f}; {t.height} lines with >= {MIN_GAMES:,} games")
        for lab, rows in (("best for White", t.tail(5).reverse()), ("worst for White", t.head(5))):
            print("  " + lab + ": " + "; ".join(f"{r['line']} {r['rel']:+.3f} (z {r['z']:+.0f})" for r in rows.to_dicts()))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "opening_perf.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
