"""Does thinking longer help, at fixed difficulty? 1500-band train table (Stockfish labels, Maia-3 difficulty; CPU).

    python -m doppel_lab.think_value

Per own move (ply >= 10, clock fraction >= 0.1 so time trouble doesn't drive it): difficulty = Maia-3 policy
entropy decile; relative think = this move's think fraction minus the side's mean think fraction in the game
(the player's own pace), in quintiles. Blunder rate (wc loss >= 0.3) and mean win-chance loss per cell. Raw
think-time quintiles are shown too: they confound difficulty (hard positions get long thinks and blunders).
Also per side: Spearman between think fraction and entropy across its moves ("time allocation"), and whether
sides that allocate more to hard positions blunder less (rating-matched: the band is 1500-1599).
Output: lab/results/think_value.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from doppel_lab.features import table

c = pl.col


def main():
    from doppel_lab.bench import RESULTS
    t = table.load("train", "T2", 1500, columns=["game_id", "mover", "ply", "think_frac", "maia_entropy", "maia_p_top",
                                                   "clock_frac", "blunder", "wc_loss", "elo_self"])
    t = t.filter((c("ply") >= 10) & (c("clock_frac") >= 0.1) & c("think_frac").is_finite() & c("maia_entropy").is_finite()
                 & c("wc_loss").is_not_null())
    t = t.with_columns(rel=c("think_frac") - c("think_frac").mean().over("game_id", "mover"))
    t = t.with_columns(dq=c("maia_entropy").qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True),
                       rq=c("rel").qcut(5, labels=["much faster", "faster", "own pace", "slower", "much slower"]),
                       tq=c("think_frac").qcut(5, labels=["q1", "q2", "q3", "q4", "q5"]))
    raw = t.group_by("tq").agg(n=pl.len(), blunder=c("blunder").mean(), wc=c("wc_loss").mean()).sort("tq")
    cell = t.group_by("dq", "rq").agg(n=pl.len(), blunder=c("blunder").mean(), wc=c("wc_loss").mean())
    piv = cell.pivot(on="rq", index="dq", values="blunder").sort("dq")
    with pl.Config(tbl_cols=8, float_precision=4):
        print("raw think-fraction quintile:"); print(raw)
        print("blunder rate by difficulty decile (rows, Maia entropy) x think relative to own pace (columns):"); print(piv)
    # pooled effect of relative think within difficulty deciles (weighted by cell size)
    w = cell.pivot(on="rq", index="dq", values="n").sort("dq")
    eff = {}
    for col in ["much faster", "faster", "slower", "much slower"]:
        d = (piv[col] - piv["own pace"]).to_numpy()
        ww = (w[col] + w["own pace"]).to_numpy()
        eff[col] = float((d * ww).sum() / ww.sum())
    print("blunder rate vs own pace, within difficulty deciles:", {k: round(v, 4) for k, v in eff.items()})
    # per-side time allocation
    side = t.group_by("game_id", "mover").agg(n=pl.len(), alloc=pl.corr("think_frac", "maia_entropy", method="spearman"),
                                              blunder=c("blunder").mean(), think=c("think_frac").mean(),
                                              ent=c("maia_entropy").mean()).filter(c("n") >= 15)
    side = side.with_columns(aq=c("alloc").qcut(5, labels=["a1", "a2", "a3", "a4", "a5"]))
    by_alloc = side.group_by("aq").agg(n=pl.len(), alloc=c("alloc").mean(), blunder=c("blunder").mean(),
                                       think=c("think").mean(), ent=c("ent").mean()).sort("aq")
    with pl.Config(float_precision=4):
        print("sides by time-allocation quintile (Spearman of think vs Maia entropy across their moves):"); print(by_alloc)
    out = {"raw_think_quintile": raw.to_dicts(), "blunder_by_difficulty_x_relthink": piv.to_dicts(),
           "relthink_effect_within_difficulty": eff, "by_allocation": by_alloc.to_dicts()}
    (RESULTS / "think_value.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
