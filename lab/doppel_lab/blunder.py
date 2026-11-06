"""Blunder labels for v0 games (docs/SCIENCE.md §7): a sidecar joined on game_id.

Per game, per ply k (the move from position k, by the side to move), from the mover's point of view:
  wc_loss   drop in Lichess winning chances (range -1..1) from Stockfish at 50k nodes, >= 0
  cpl       centipawn loss, evals clipped to +/-1000
  blunder   wc_loss >= 0.3 (Lichess's blunder)
  blunder2  wc_loss >= 0.2 (Maia's unit: 10 points of win probability)

Steps (CPU; run the Stockfish pass at low priority so a training job's data loader keeps its cores):
    python -m doppel_lab.blunder select --split train --games 100000     # -> data/v0/blunder_games_1500_train.parquet
    python -m doppel_lab.blunder select --split val                       # all val games
    nice -n 15 python -m doppel_lab.analyze data/v0/blunder_games_1500_train.parquet \
        data/v0/evals_1500_train.parquet --workers 24                     # ~3 h for 100k games; resumable
    python -m doppel_lab.blunder label --split train
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from doppel_lab.resign import V0

THRESH = 0.3
THRESH_MAIA = 0.2


def labels(evals) -> dict:
    """evals[k] is White-POV cp of position k (evals[0] = start position); returns per-ply mover-POV labels.
    Same definitions as the registry's T2 features (doppel_lab.features.channels.eval_channels)."""
    from doppel_lab.features.channels import eval_channels

    n = len(evals) - 1
    mov = {"ply": np.arange(n), "mover": np.arange(n) % 2}
    eval_channels({}, mov, evals)
    wl = mov["wc_loss"]
    return {"wc_loss": wl.astype(np.float32), "cpl": mov["cpl"].astype(np.float32),
            "blunder": (wl >= THRESH).astype(np.int8), "blunder2": (wl >= THRESH_MAIA).astype(np.int8)}


def select(split: str, band: int, n_games: int | None, seed: int = 0):
    g = pl.read_parquet(V0 / f"games_{band}_{split}.parquet", columns=["game_id", "moves"])
    if n_games and n_games < g.height:
        g = g.sample(n_games, seed=seed, shuffle=True)
    out = V0 / f"blunder_games_{band}_{split}.parquet"
    g.write_parquet(out)
    print(f"{split}: {g.height:,} games, {g['moves'].list.len().sum():,} positions -> {out}")


def label(split: str, band: int):
    ev = pl.read_parquet(V0 / f"evals_{band}_{split}.parquet")
    games = pl.read_parquet(V0 / f"blunder_games_{band}_{split}.parquet", columns=["game_id", "moves"])
    ev = ev.join(games, on="game_id")
    rows = []
    for gid, evals, moves in ev.select("game_id", "evals", "moves").iter_rows():
        assert len(evals) == len(moves) + 1, gid
        rows.append({"game_id": gid, **{k: v.tolist() for k, v in labels(evals).items()}})
    out = pl.DataFrame(rows)
    path = V0 / f"blunder_{band}_{split}.parquet"
    out.write_parquet(path)
    b = np.concatenate(out["blunder"].to_list())
    b2 = np.concatenate(out["blunder2"].to_list())
    print(f"{split}: {out.height:,} games, {len(b):,} moves; blunder rate {b.mean():.2%} (>=0.3), "
          f"{b2.mean():.2%} (>=0.2) -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--split", choices=["train", "val"], default="val")
    s.add_argument("--band", type=int, default=1500)
    s.add_argument("--games", type=int, default=None)
    s.add_argument("--seed", type=int, default=0)
    lb = sub.add_parser("label")
    lb.add_argument("--split", choices=["train", "val"], default="val")
    lb.add_argument("--band", type=int, default=1500)
    a = ap.parse_args()
    if a.cmd == "select":
        select(a.split, a.band, a.games, a.seed)
    else:
        label(a.split, a.band)
