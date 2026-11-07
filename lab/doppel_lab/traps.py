"""Opening traps 1500 players fall into: the most common early blunders by position (train games with evals, CPU).

    python -m doppel_lab.traps

100k 1500-band train games with 50k-node evals. Own moves at plies 4..24; blunder = wc loss >= 0.3. Grouped by
(move-order prefix, move played). Listed: the (position, blunder) pairs with the most occurrences, with the
position's visit count and how often it was blundered there. SAN for readability. Candidate lesson content.
Output: lab/results/traps.json
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import polars as pl

from doppel_lab.features.channels import wc

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "data" / "v0"
RESULTS = ROOT / "lab" / "results"
c = pl.col


def san_line(moves: list[str]) -> str:
    b, out = chess.Board(), []
    for i, u in enumerate(moves):
        m = chess.Move.from_uci(u)
        out.append((f"{i // 2 + 1}." if i % 2 == 0 else "") + b.san(m))
        b.push(m)
    return " ".join(out)


def main():
    g = pl.read_parquet(V0 / "blunder_games_1500_train.parquet").join(pl.read_parquet(V0 / "evals_1500_train.parquet"), on="game_id")
    rows = []
    for moves, ev in zip(g["moves"].to_list(), g["evals"].to_list()):
        w = wc(np.asarray(ev, float))
        for k in range(4, min(25, len(moves), len(w) - 1)):
            sgn = 1 if k % 2 == 0 else -1
            rows.append((" ".join(moves[:k]), moves[k], bool(sgn * (w[k] - w[k + 1]) >= 0.3)))
    d = pl.DataFrame(rows, orient="row", schema=["pre", "mv", "bl"])
    visits = d.group_by("pre").len().rename({"len": "visits"})
    t = (d.filter(c("bl")).group_by("pre", "mv").len().join(visits, on="pre")
         .with_columns(share=c("len") / c("visits")).sort("len", descending=True).head(25))
    out = []
    for r in t.to_dicts():
        pre = r["pre"].split(" ")
        line = san_line(pre + [r["mv"]])
        out.append({"line": line, "times": r["len"], "visits": r["visits"], "blunder_share_here": r["share"]})
        print(f"{r['len']:5d}x of {r['visits']:6d} visits ({r['share']:.0%}): {line}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "traps.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
