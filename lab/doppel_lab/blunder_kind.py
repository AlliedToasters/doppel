"""What kind of move is a blunder? Move types vs blunder rate, 1500-band train games with evals (CPU).

    python -m doppel_lab.blunder_kind [--games 30000]

Replays games with python-chess; per own move at ply >= 10: piece moved, capture, check given, the move
retreats a piece that was attacked (escape), the moved piece lands on a square attacked by the opponent and
not defended (hangs, a crude static test), and whether the opponent's last move was a capture (recapture
chance). Blunder = wc loss >= 0.3 (the label rule, from stored 50k-node evals). Rates and lifts per type.
Output: lab/results/blunder_kind.json
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import chess
import numpy as np
import polars as pl

from doppel_lab.features.channels import wc

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "data" / "v0"
RESULTS = ROOT / "lab" / "results"
NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen",
         chess.KING: "king"}


def game(task):
    moves, ev = task
    w = wc(np.asarray(ev, float))
    b = chess.Board()
    rows = []
    prev_cap = False
    for k, u in enumerate(moves):
        m = chess.Move.from_uci(u)
        if k >= 10 and k + 1 < len(w):
            me = b.turn
            sgn = 1 if me == chess.WHITE else -1
            loss = sgn * (w[k] - w[k + 1])
            pt = b.piece_type_at(m.from_square)
            cap = b.is_capture(m)
            was_att = b.is_attacked_by(not me, m.from_square)
            gives = b.gives_check(m)
            b.push(m)
            hang = pt != chess.KING and b.is_attacked_by(not me, m.to_square) and not b.is_attacked_by(me, m.to_square)
            rows.append((NAMES.get(pt, "?"), cap, gives, was_att and pt != chess.PAWN, hang, prev_cap, loss >= 0.3))
            prev_cap = cap
        else:
            prev_cap = b.is_capture(m)
            b.push(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30_000)
    a = ap.parse_args()
    g = pl.read_parquet(V0 / "blunder_games_1500_train.parquet").join(pl.read_parquet(V0 / "evals_1500_train.parquet"),
                                                                       on="game_id").head(a.games)
    with Pool(20) as p:
        out = p.map(game, list(zip(g["moves"].to_list(), g["evals"].to_list())), chunksize=50)
    d = pl.DataFrame([r for rs in out for r in rs], orient="row",
                     schema=["piece", "capture", "check", "escape", "hangs", "after_capture", "blunder"])
    base = float(d["blunder"].mean())
    res = {"moves": d.height, "base_rate": base, "groups": {}}
    print(f"{d.height:,} moves, blunder rate {base:.4f}")
    for col in ["piece", "capture", "check", "escape", "hangs", "after_capture"]:
        t = d.group_by(col).agg(n=pl.len(), rate=c_("blunder").mean()).sort(col)
        res["groups"][col] = t.to_dicts()
        print(f"  {col:13s} " + "  ".join(f"{r[col]}: {r['rate']:.4f} (x{r['rate'] / base:.2f}, {r['n'] / d.height:.0%})"
                                         for r in t.to_dicts()))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "blunder_kind.json").write_text(json.dumps(res, indent=1, default=str) + "\n")


c_ = pl.col

if __name__ == "__main__":
    main()
