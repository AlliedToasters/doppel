"""When do 1500 players resign, and is it premature? (train games with stored Stockfish evals; CPU)

    python -m doppel_lab.resign_when

100k train games (1500 band) with per-ply 50k-node evals. Resignations: the resigner's win chances (Lichess wc,
-1..1) at the last position. Rationality check: among all positions where a side stood at win chances in a
bin and the game went on, how often that side went on to draw or win (salvage rate). If players resign at
positions they would salvage often, the resignation is premature.
Output: lab/results/resign_when.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.features.channels import wc

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "data" / "v0"
RESULTS = ROOT / "lab" / "results"
BINS = [-1.01, -0.95, -0.8, -0.6, -0.4, -0.2, 0.0]
c = pl.col


def main():
    ev = pl.read_parquet(V0 / "evals_1500_train.parquet")
    g = pl.read_parquet(V0 / "games_1500_train.parquet", columns=["game_id", "result", "clocks", "base"])
    r = pl.read_parquet(V0 / "resign_1500_train.parquet")
    d = ev.join(g, on="game_id").join(r, on="game_id")
    print(f"{d.height:,} games with evals; ends: {dict(d['end'].value_counts().iter_rows())}", flush=True)
    res_wc, res_clock, salv_x, salv_y = [], [], [], []
    rng = np.random.default_rng(0)
    for e, result, clocks, base, end, who in zip(d["evals"].to_list(), d["result"].to_list(), d["clocks"].to_list(),
                                                d["base"].to_list(), d["end"].to_list(), d["resigner"].to_list()):
        w = wc(np.asarray(e, float))
        score_w = 1.0 if result == "1-0" else 0.0 if result == "0-1" else 0.5
        if end == "resign" and who in ("w", "b"):
            sgn = 1 if who == "w" else -1
            res_wc.append(sgn * w[-1])
            side_clocks = clocks[0 if who == "w" else 1::2]
            res_clock.append(side_clocks[-1] / base if side_clocks and base else np.nan)
        # salvage: sample up to 4 positions per game, ply >= 10, both sides' view, excluding the last 2 plies
        n = len(w)
        if n > 14:
            for k in rng.choice(np.arange(10, n - 2), min(4, n - 12), replace=False):
                for sgn, sc in ((1, score_w), (-1, 1 - score_w)):
                    salv_x.append(sgn * w[k])
                    salv_y.append(sc >= 0.5)
    rw, rc = np.array(res_wc), np.array(res_clock)
    sx, sy = np.array(salv_x), np.array(salv_y)
    out = {"resignations": len(rw), "bins": []}
    print(f"{len(rw):,} resignations; resigner's win chances: median {np.median(rw):+.3f}, "
          f"share above -0.6 {np.mean(rw > -0.6):.3f}, above -0.2 {np.mean(rw > -0.2):.3f}; "
          f"median clock left {np.nanmedian(rc):.2f} of base", flush=True)
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m, ms = (rw > lo) & (rw <= hi), (sx > lo) & (sx <= hi)
        row = {"wc_lo": lo, "wc_hi": hi, "resign_share": float(m.mean()), "salvage_rate_if_played_on": float(sy[ms].mean()),
               "positions": int(ms.sum())}
        out["bins"].append(row)
        print(f"  wc ({lo:+.2f}, {hi:+.2f}]: {row['resign_share']:.3f} of resignations; side standing here "
              f"later draws or wins {row['salvage_rate_if_played_on']:.3f} (n {row['positions']:,})", flush=True)
    out["median_resign_wc"] = float(np.median(rw))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "resign_when.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
