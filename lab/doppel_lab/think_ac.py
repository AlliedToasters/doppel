"""Do long thinks cluster? Autocorrelation of a side's think times after difficulty and clock (train table, CPU).

    python -m doppel_lab.think_ac

1500-band train table, own moves at ply >= 10. y = log(think_s + 0.5). Residual after a linear fit on Maia-3
entropy, top-move probability, clock fraction (and its log), ply and the side's game mean (pace); then the
correlation of a side's residual with its own residual 1, 2, 3, 5, 10 own moves earlier. A positive lag-1
residual correlation is state the think head needs to carry (an "in a think" or "in a hurry" mode).
Output: lab/results/think_ac.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from doppel_lab.features import table

LAGS = (1, 2, 3, 5, 10)
c = pl.col


def main():
    from doppel_lab.bench import RESULTS
    t = table.load("train", "T2", 1500, columns=["game_id", "mover", "ply", "think_s", "maia_entropy", "maia_p_top",
                                                   "clock_frac"]).sort("game_id", "ply")
    t = t.filter((c("ply") >= 10) & c("think_s").is_finite() & c("maia_entropy").is_finite() & c("clock_frac").is_finite())
    t = t.with_columns(y=(c("think_s") + 0.5).log(), lcf=(c("clock_frac") + 0.01).log())
    t = t.with_columns(pace=c("y").mean().over("game_id", "mover"))
    X = np.c_[np.ones(t.height), t.select("maia_entropy", "maia_p_top", "clock_frac", "lcf", "ply", "pace").to_numpy()]
    y = t["y"].to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    r2 = 1 - r.var() / y.var()
    t = t.with_columns(r=pl.Series(r))
    out = {"rows": t.height, "r2_of_controls": float(r2), "lags": {}}
    print(f"{t.height:,} moves; controls explain R² {r2:.3f} of log think", flush=True)
    for L in LAGS:
        u = t.with_columns(rl=c("r").shift(L).over("game_id", "mover"), yl=c("y").shift(L).over("game_id", "mover")).drop_nulls(["rl"])
        rc = float(np.corrcoef(u["r"], u["rl"])[0, 1])
        raw = float(np.corrcoef(u["y"] - u["pace"], u["yl"] - u["pace"])[0, 1])
        out["lags"][str(L)] = {"resid_corr": rc, "raw_corr_within_game": raw}
        print(f"lag {L:2d}: residual corr {rc:+.3f}   raw within-game corr {raw:+.3f}", flush=True)
    (RESULTS / "think_ac.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
