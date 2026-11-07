"""Opening fashion by year and tier, online (Lichess March heads 2013-2026) and OTB (TWIC classical). CPU.

    python -m doppel_lab.fashion

A natural experiment: Netflix's "The Queen's Gambit" aired October 2020, between the 2020-03 and 2021-03 samples.
Lines are matched on the first plies (UCI, so move-order-specific):
  queens_gambit  1.d4 d5 2.c4         london   1.d4 and Bf4 among White's first 3 moves (no c4)
  sicilian       1.e4 c5               french   1.e4 e6        caro_kann  1.e4 c6       scandi  1.e4 d5
  italian        1.e4 e5 2.Nf3 Nc6 3.Bc4                      ruy_lopez  1.e4 e5 2.Nf3 Nc6 3.Bb5
  kings_indian   1.d4 Nf6 2.c4 g6     english  1.c4             reti      1.Nf3
Shares are of all games in the group. Lichess tiers are within-year percentiles; TWIC tiers are FIDE bands.
Output: lab/results/fashion.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
LAKE = ROOT / "data" / "lake"
RESULTS = ROOT / "lab" / "results"
c = pl.col


def lines(m: pl.Expr) -> dict[str, pl.Expr]:
    g = lambda i: m.list.get(i, null_on_oob=True)
    w3 = pl.concat_list([g(0), g(2), g(4)])
    return {
        "queens_gambit": (g(0) == "d2d4") & (g(1) == "d7d5") & (g(2) == "c2c4"),
        "london": (g(0) == "d2d4") & w3.list.contains("c1f4") & ~w3.list.contains("c2c4"),
        "sicilian": (g(0) == "e2e4") & (g(1) == "c7c5"),
        "french": (g(0) == "e2e4") & (g(1) == "e7e6"),
        "caro_kann": (g(0) == "e2e4") & (g(1) == "c7c6"),
        "scandi": (g(0) == "e2e4") & (g(1) == "d7d5"),
        "italian": (g(0) == "e2e4") & (g(1) == "e7e5") & (g(2) == "g1f3") & (g(3) == "b8c6") & (g(4) == "f1c4"),
        "ruy_lopez": (g(0) == "e2e4") & (g(1) == "e7e5") & (g(2) == "g1f3") & (g(3) == "b8c6") & (g(4) == "f1b5"),
        "kings_indian": (g(0) == "d2d4") & (g(1) == "g8f6") & (g(2) == "c2c4") & (g(3) == "g7g6"),
        "english": g(0) == "c2c4",
        "reti": g(0) == "g1f3",
    }


def shares(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    L = lines(c("moves"))
    return df.with_columns(**L).group_by(by).agg(n=pl.len(), **{k: c(k).mean() for k in L}).sort(by)


def main():
    res = {}
    parts = []
    for p in sorted((LAKE / "lichess").glob("????-??")):
        d = pl.read_parquet(p / "*.parquet", columns=["moves", "white_elo", "black_elo", "fen", "plies"]).filter(
            c("fen").is_null() & c("white_elo").is_not_null() & c("black_elo").is_not_null() & (c("plies") >= 6))
        d = d.with_columns(year=pl.lit(int(p.name[:4])), elo=(c("white_elo") + c("black_elo")) / 2)
        d = d.with_columns(pct=c("elo").rank() / pl.len() * 100)
        parts.append(d.with_columns(tier=pl.when(c("pct") < 25).then(pl.lit("p00-25"))
                                    .when(c("pct") < 75).then(pl.lit("p25-75"))
                                    .when(c("pct") < 95).then(pl.lit("p75-95")).otherwise(pl.lit("p95+"))))
    li = pl.concat(parts)
    on_all, on_tier = shares(li, ["year"]), shares(li, ["year", "tier"])
    from doppel_lab.twic_science import load
    tw = load().filter((c("medium") == "otb") & (c("speed") == "classical") & (c("age") == "open")
                       & c("elo").is_not_null() & (c("plies") >= 6))
    tw = tw.with_columns(tier=pl.when(c("elo") < 2200).then(pl.lit("<2200")).when(c("elo") < 2400)
                         .then(pl.lit("2200-2399")).when(c("elo") < 2600).then(pl.lit("2400-2599"))
                         .otherwise(pl.lit("2600+")))
    otb_all, otb_tier = shares(tw, ["year"]), shares(tw, ["year", "tier"])
    cols = ["queens_gambit", "london", "sicilian", "caro_kann", "scandi", "italian", "ruy_lopez", "english"]
    with pl.Config(tbl_rows=40, tbl_cols=14, float_precision=3, tbl_width_chars=220):
        print("ONLINE (Lichess, all games)")
        print(on_all.select("year", "n", *cols))
        print("ONLINE top 5% only")
        print(on_tier.filter(c("tier") == "p95+").select("year", "n", *cols))
        print("OTB (TWIC classical open)")
        print(otb_all.select("year", "n", *cols))
        print("OTB 2600+")
        print(otb_tier.filter(c("tier") == "2600+").select("year", "n", *cols))
    # Queen's Gambit natural experiment: 2020-03 -> 2021-03, per tier, with a binomial z.
    qg = {}
    for t in ["p00-25", "p25-75", "p75-95", "p95+"]:
        a = on_tier.filter((c("year") == 2020) & (c("tier") == t)).row(0, named=True)
        b = on_tier.filter((c("year") == 2021) & (c("tier") == t)).row(0, named=True)
        p0, p1 = a["queens_gambit"], b["queens_gambit"]
        se = np.sqrt(p0 * (1 - p0) / a["n"] + p1 * (1 - p1) / b["n"])
        qg[t] = {"2020": p0, "2021": p1, "ratio": p1 / p0, "z": (p1 - p0) / se}
    print("Queen's Gambit share 2020-03 -> 2021-03 by tier:", json.dumps(qg, indent=None))
    res = {"online_by_year": on_all.to_dicts(), "online_by_year_tier": on_tier.to_dicts(),
           "otb_by_year": otb_all.to_dicts(), "otb_by_year_tier": otb_tier.to_dicts(), "queens_gambit_2020_2021": qg}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "fashion.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
