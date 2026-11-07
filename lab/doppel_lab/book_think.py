"""Think time around the book exit: do players play from memory until the line is new? (1500 train table, CPU)

    python -m doppel_lab.book_think

Per game, the book exit ply (first prefix with < 2 games in the 2026-10 train book; the train games themselves are
in that book, so the threshold is < 3 here: the game itself plus one other). Own-move think time (s) and Maia-3
entropy by ply relative to the exit (-6..+6), for exits between plies 8 and 30, so the opening's own clock
pattern doesn't dominate. If players play from memory, think jumps at the exit beyond what entropy predicts.
Output: lab/results/book_think.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from doppel_lab.book import move_ids, prefix_hashes
from doppel_lab.features import table
from doppel_lab.persistence import BOOK, RESULTS

c = pl.col


def main():
    from pathlib import Path
    V0 = Path(__file__).resolve().parents[2] / "data" / "v0"
    g = pl.read_parquet(V0 / "blunder_games_1500_train.parquet")
    ref_k, ref_c = np.load(BOOK / "table_keys.npy"), np.load(BOOK / "table_counts.npy")
    ids, off = move_ids(g["moves"])
    keys, cnt = prefix_hashes(ids, off, cap=40)
    i = np.searchsorted(ref_k, keys).clip(max=len(ref_k) - 1)
    m = np.where(ref_k[i] == keys, ref_c[i], 0)
    o = np.zeros(len(cnt) + 1, np.int64)
    np.cumsum(cnt, out=o[1:])
    start = o[:-1]
    pos = np.arange(len(keys)) - np.repeat(start, cnt)
    big = np.iinfo(np.int64).max
    f = np.minimum.reduceat(np.where(m < 3, pos, big), start)
    ex = pl.DataFrame({"game_id": g["game_id"], "exit": np.where(f == big, -1, f)}).filter(c("exit").is_between(8, 30))
    t = table.load("train", "T2", 1500, columns=["game_id", "ply", "think_s", "maia_entropy"])
    t = t.join(ex, on="game_id")   # both 0-based: ply k is moves[k], exit is the index of the leaving move
    t = t.with_columns(rel=c("ply") - c("exit"))
    t = t.filter(c("rel").is_between(-6, 6) & c("think_s").is_finite())
    r = t.group_by("rel").agg(n=pl.len(), think=c("think_s").mean(), think_med=c("think_s").median(),
                              ent=c("maia_entropy").mean()).sort("rel")
    with pl.Config(tbl_rows=20, float_precision=3):
        print(f"{ex.height:,} games with a book exit at plies 8-30; table ply range {t['ply'].min()}..{t['ply'].max()}")
        print(r)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "book_think.json").write_text(json.dumps({"games": ex.height, "by_rel": r.to_dicts()}, indent=1) + "\n")


if __name__ == "__main__":
    main()
