"""Maia-3's rating scale against today's ratings: the input rating that best explains each side's moves, by
real rating and year (Lichess bench suite + maia3_sweep.npz; CPU).

    python -m doppel_lab.maia_offset

Per side: mean played-move log-prob with Maia-3 conditioned on each of ELOS (both players at that rating). The
best input rating is the argmax over ELOS of the log-prob summed over sides in a group, refined with a quadratic
through the top three grid points (pooled curves are smooth; per-side argmax is too noisy at 200-point steps).
Groups: the side's real rating (100-point bins) x era (2018-2020, 2021-2023, 2024-2026). Offset = best input
minus the group's mean real rating.
Output: lab/results/maia_offset.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from doppel_lab.bench import BENCH
from doppel_lab.bench_embed import ELOS

RESULTS = BENCH.parents[1] / "lab" / "results"
c = pl.col


def best(curve: np.ndarray) -> float:
    i = int(np.argmax(curve))
    if 0 < i < len(curve) - 1:
        y0, y1, y2 = curve[i - 1], curve[i], curve[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(ELOS[i] + 0.5 * (y0 - y2) / den * (ELOS[1] - ELOS[0]))
    return float(ELOS[i])


def main():
    z = np.load(BENCH / "maia3_sweep.npz")
    t = pl.read_parquet(BENCH / "tasks.parquet", columns=["game_id", "white_elo", "black_elo", "year", "in_main"])
    gi = {g: i for i, g in enumerate(z["game_ids"])}
    t = t.filter(c("game_id").is_in(list(gi))).with_columns(i=c("game_id").replace_strict(gi, return_dtype=pl.Int64))
    ll = z["ll"]
    rows = []
    for col, k in (("white_elo", 0), ("black_elo", 1)):
        x = t.select("i", "year", elo=c(col))
        L = ll[x["i"].to_numpy(), k]
        rows.append(x.with_columns(pl.Series("ll", list(L))))
    s = pl.concat(rows).filter(c("ll").list.eval(pl.element().is_finite()).list.all())
    s = s.with_columns(bin=(c("elo") // 100 * 100).clip(700, 2600),
                       era=pl.when(c("year") <= 2020).then(pl.lit("2018-20")).when(c("year") <= 2023).then(pl.lit("2021-23"))
                       .otherwise(pl.lit("2024-26")))
    res = {"elos": list(ELOS), "groups": []}
    for (b, era), sub in sorted(s.group_by("bin", "era"), key=lambda kv: (kv[0][1], kv[0][0])):
        if sub.height < 300:
            continue
        curve = np.mean(np.stack(sub["ll"].to_list()), axis=0)
        bst = best(curve)
        r = {"bin": int(b), "era": era, "sides": sub.height, "mean_elo": float(sub["elo"].mean()), "best_input": bst,
             "offset": bst - float(sub["elo"].mean())}
        res["groups"].append(r)
    for era in ("2018-20", "2021-23", "2024-26"):
        print(era + ": " + "  ".join(f"{r['bin']}:{r['offset']:+.0f}" for r in res["groups"] if r["era"] == era), flush=True)
    (RESULTS / "maia_offset.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
