"""Playing on for the flag: in a lost position, how much does a clock lead raise the chance of a save? (CPU)

    python -m doppel_lab.flag_hope

Same 100k train games as resign_when.py. Positions at ply >= 10 where the side to judge stands at win chances
<= -0.6 (lost). Clock lead = own clock minus the opponent's, as a fraction of base, at that ply (clocks are
recorded after each move, so the pair is the last two). Save = the side later draws or wins; by clock-lead
bin, and how the save happens (opponent flagged, opponent blundered back into a draw/loss on the board).
Output: lab/results/flag_hope.json
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
LEADS = [-1.0, -0.3, -0.1, -0.03, 0.03, 0.1, 0.3, 1.0]


def main():
    ev = pl.read_parquet(V0 / "evals_1500_train.parquet")
    g = pl.read_parquet(V0 / "games_1500_train.parquet", columns=["game_id", "result", "termination", "clocks", "base"])
    d = ev.join(g, on="game_id")
    rng = np.random.default_rng(0)
    rows = []
    for e, result, term, clocks, base in zip(d["evals"].to_list(), d["result"].to_list(), d["termination"].to_list(),
                                             d["clocks"].to_list(), d["base"].to_list()):
        if not clocks or not base or len(clocks) + 1 != len(e):
            continue
        w = wc(np.asarray(e, float))
        cl = np.asarray(clocks, float)
        score_w = 1.0 if result == "1-0" else 0.0 if result == "0-1" else 0.5
        n = len(cl)
        ks = [k for k in range(10, n - 1)]
        if not ks:
            continue
        for k in rng.choice(ks, min(3, len(ks)), replace=False):
            # after ply k (1-based count k+1 moves): position index k+1; clocks[k] is the mover of ply k
            pos = k + 1
            mover_white = k % 2 == 0
            cw = cl[k] if mover_white else cl[k - 1]
            cb = cl[k - 1] if mover_white else cl[k]
            for side, sgn in (("w", 1), ("b", -1)):
                v = sgn * w[pos]
                if v > -0.6:
                    continue
                lead = ((cw - cb) if side == "w" else (cb - cw)) / base
                sc = score_w if side == "w" else 1 - score_w
                rows.append((lead, sc >= 0.5, sc == 1 and term == "Time forfeit", sc >= 0.5 and term != "Time forfeit"))
    a = np.array(rows, float)
    out = {"positions": len(a), "bins": []}
    print(f"{len(a):,} lost positions (wc <= -0.6); overall save rate {a[:, 1].mean():.3f}", flush=True)
    for lo, hi in zip(LEADS[:-1], LEADS[1:]):
        m = (a[:, 0] > lo) & (a[:, 0] <= hi)
        r = {"lead_lo": lo, "lead_hi": hi, "n": int(m.sum()), "save": float(a[m, 1].mean()),
             "win_on_time": float(a[m, 2].mean()), "save_on_board": float(a[m, 3].mean())}
        out["bins"].append(r)
        print(f"  clock lead ({lo:+.2f}, {hi:+.2f}] of base: n {r['n']:6,}  saved {r['save']:.3f}  "
              f"(won on time {r['win_on_time']:.3f}, drew/won on board {r['save_on_board']:.3f})", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "flag_hope.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
