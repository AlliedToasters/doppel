"""Does think time change *which* move is played? Maia-3 agreement by think time at fixed difficulty (CPU).

    python -m doppel_lab.think_policy

1500-band train table, own moves at ply >= 10, clock >= 10% of base. Played move's Maia-3 rank (1 = Maia's top
choice) and log-prob, by think bucket inside Maia entropy quintiles. If long thinks push humans away from
Maia's intuitive top move (or toward it), a Doppel's policy should be conditioned on its own sampled think time.
Output: lab/results/think_policy.json
"""

from __future__ import annotations

import json

import polars as pl

from doppel_lab.features import table

c = pl.col


def main():
    from doppel_lab.bench import RESULTS
    t = table.load("train", "T2", 1500, columns=["ply", "think_s", "clock_frac", "maia_entropy", "maia_played_rank",
                                                   "maia_played_lp", "blunder"])
    t = t.filter((c("ply") >= 10) & (c("clock_frac") >= 0.1) & c("think_s").is_finite() & c("maia_entropy").is_finite())
    t = t.with_columns(tb=pl.when(c("think_s") < 0.5).then(pl.lit("a 0s")).when(c("think_s") < 1.5).then(pl.lit("b 1s"))
                       .when(c("think_s") < 3.5).then(pl.lit("c 2-3s")).when(c("think_s") < 10.5).then(pl.lit("d 4-10s"))
                       .otherwise(pl.lit("e 11s+")),
                       dq=c("maia_entropy").qcut(5, labels=["e1", "e2", "e3", "e4", "e5"]),
                       top1=(c("maia_played_rank") == 1).cast(pl.Float64))
    top1 = t.group_by("dq", "tb").agg(v=c("top1").mean()).pivot(on="tb", index="dq", values="v", sort_columns=True).sort("dq")
    lp = t.group_by("dq", "tb").agg(v=c("maia_played_lp").mean()).pivot(on="tb", index="dq", values="v", sort_columns=True).sort("dq")
    # among non-blunders only: is a slow move a *different good* move or just a worse one?
    nb = t.filter(~c("blunder")).group_by("dq", "tb").agg(v=c("top1").mean()).pivot(on="tb", index="dq", values="v",
                                                                                      sort_columns=True).sort("dq")
    with pl.Config(float_precision=3, tbl_cols=8):
        print("played = Maia-3 top-1, by entropy quintile x think:"); print(top1)
        print("mean played-move log-prob:"); print(lp)
        print("top-1 agreement among non-blunders:"); print(nb)
    (RESULTS / "think_policy.json").write_text(json.dumps({"top1": top1.to_dicts(), "lp": lp.to_dicts(),
                                                           "top1_nonblunder": nb.to_dicts()}, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
