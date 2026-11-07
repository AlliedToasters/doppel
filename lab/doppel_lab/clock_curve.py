"""Time management by rating: share of the base clock left at each move number (2026-10 corpus, 3+0, CPU).

    python -m doppel_lab.clock_curve

Per side, the clock after own moves 10, 20, 30, 40, 50 (only games that reach that move), as a fraction of the
180 s base; median and the share of sides under 10% (time trouble), by the side's rating (200-point bands).
Survivorship: later moves only count games still going, so long games weigh more there.
Output: lab/results/clock_curve.json
"""

from __future__ import annotations

import json

import polars as pl

from doppel_lab.scramble import CORPUS, RESULTS

MOVES = (10, 20, 30, 40, 50)
c = pl.col


def main():
    d = pl.scan_parquet(str(CORPUS / "*" / "*.parquet")).filter((c("base") == 180) & (c("inc") == 0)).select(
        "white_elo", "black_elo", "clocks").collect()
    parts = []
    for side, col in ((0, "white_elo"), (1, "black_elo")):
        x = d.select(elo=c(col), own=c("clocks").list.gather_every(2, offset=side))
        for mv in MOVES:
            parts.append(x.filter(c("own").list.len() >= mv).select(
                band=(c("elo") // 200 * 200).clip(600, 2600), move=pl.lit(mv), left=c("own").list.get(mv - 1) / 180))
    t = pl.concat(parts).group_by("band", "move").agg(med=c("left").median(), trouble=(c("left") < 0.1).mean(), n=pl.len())
    med = t.pivot(on="move", index="band", values="med", sort_columns=True).sort("band")
    tr = t.pivot(on="move", index="band", values="trouble", sort_columns=True).sort("band")
    with pl.Config(float_precision=3, tbl_rows=20):
        print("median share of base left after own move N:"); print(med)
        print("share of sides under 10% of base:"); print(tr)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "clock_curve.json").write_text(json.dumps({"median_left": med.to_dicts(), "trouble": tr.to_dicts()},
                                                         indent=1) + "\n")


if __name__ == "__main__":
    main()
