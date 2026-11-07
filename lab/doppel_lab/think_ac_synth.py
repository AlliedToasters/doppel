"""Do synthetic players reproduce human think-time spells? Within-game autocorrelation of log think (CPU).

    python -m doppel_lab.think_ac_synth

Human 1500 val games vs self-play games (v0d Doppel, ChessMimic 1500; Maia-3 has no clock model) at 3+0.\nv0d self-play records no clocks (its think head is scored on human games), so it is skipped until it does.
Per side, own moves from the 6th on: y = log(think + 0.5) minus the side's game mean; Pearson correlation of y
with the side's own y at lag 1, 2, 3, 5. Raw (no difficulty control) on every source, so they're comparable.
Output: lab/results/think_ac_synth.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"human": ROOT / "data" / "v0" / "games_1500_val.parquet",
           "doppel_v0d": ROOT / "data" / "synth" / "doppel_v0d_1500.parquet",
           "chessmimic": ROOT / "data" / "synth" / "chessmimic_1500.parquet"}
LAGS = (1, 2, 3, 5)
c = pl.col


def lagcorr(df: pl.DataFrame, n_max: int = 20_000) -> dict:
    df = df.filter((c("base") == 180) & (c("inc") == 0)).head(n_max)
    ys = []
    for clocks in df["clocks"].to_list():
        if not clocks:
            continue
        for side in (0, 1):
            a = np.asarray(clocks[side::2], float)
            th = np.clip(a[:-1] - a[1:], 0, 300)[4:]
            if len(th) >= 12:
                y = np.log(th + 0.5)
                ys.append(y - y.mean())
    out = {"sides": len(ys)}
    if not ys:
        return out
    for L in LAGS:
        x = np.concatenate([y[L:] for y in ys])
        z = np.concatenate([y[:-L] for y in ys])
        out[str(L)] = float(np.corrcoef(x, z)[0, 1])
    return out


def main():
    res = {}
    for name, p in SOURCES.items():
        if not p.exists():
            continue
        res[name] = lagcorr(pl.read_parquet(p, columns=["base", "inc", "clocks"]))
        if not res[name]["sides"]:
            print(f"{name:11s} no clocks recorded", flush=True)
            continue
        print(f"{name:11s} sides {res[name]['sides']:6,}  " + "  ".join(f"lag {L} {res[name][str(L)]:+.3f}" for L in LAGS), flush=True)
    (ROOT / "lab" / "results" / "think_ac_synth.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
