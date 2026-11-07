"""Are instant moves (premoves) a separate blunder risk? Blunder rate by think bucket x difficulty (train table, CPU).

    python -m doppel_lab.premove

Own moves at ply >= 10, 1500-band train table. Think buckets on Lichess's whole-second clocks: 0 s (premoves and
instant replies), 1 s, 2-3 s, 4-10 s, 11 s+. Cross-tabbed with Maia-3 entropy quintiles and clock fraction.
Output: lab/results/premove.json
"""

from __future__ import annotations

import json

import polars as pl

from doppel_lab.features import table

c = pl.col


def main():
    from doppel_lab.bench import RESULTS
    t = table.load("train", "T2", 1500, columns=["ply", "think_s", "clock_frac", "maia_entropy", "blunder", "maia_p_top"])
    t = t.filter((c("ply") >= 10) & c("think_s").is_finite() & c("maia_entropy").is_finite())
    t = t.with_columns(tb=pl.when(c("think_s") < 0.5).then(pl.lit("a 0s")).when(c("think_s") < 1.5).then(pl.lit("b 1s"))
                       .when(c("think_s") < 3.5).then(pl.lit("c 2-3s")).when(c("think_s") < 10.5).then(pl.lit("d 4-10s"))
                       .otherwise(pl.lit("e 11s+")),
                       dq=c("maia_entropy").qcut(5, labels=["e1", "e2", "e3", "e4", "e5"]))
    a = t.group_by("tb").agg(n=pl.len(), blunder=c("blunder").mean(), p_top=c("maia_p_top").mean()).sort("tb")
    b = t.group_by("dq", "tb").agg(blunder=c("blunder").mean()).pivot(on="tb", index="dq", values="blunder",
                                                                      sort_columns=True).sort("dq")
    with pl.Config(float_precision=4, tbl_cols=8):
        print(a); print(b)
    (RESULTS / "premove.json").write_text(json.dumps({"by_think": a.to_dicts(), "by_difficulty": b.to_dicts()},
                                                     indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
