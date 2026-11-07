"""Band-specific books: do strong players follow strong players' lines, or just the popular ones? (CPU)

    python -m doppel_lab.bandbook

The 2026-10 book (book.py) counts every band together, and the corpus is mostly 1000-1800, so "high-rated
games stay in book longer" could mean they follow the crowd's lines. Here each rating group gets its own book
built from the same number of train games (N_BOOK, equal so depth compares), and held-out games of every
group are scored against every group's book.
Per (eval group, book group): mean T1 / T10 exit ply, weird-move rate (a move chosen by < 1% of the >= 100
book games reaching that prefix), and the mean log10 conditional frequency q of the moves played in plies
1-12 (higher = more expected). A diagonal that beats its row means players follow their own band's lines.
Output: lab/results/bandbook.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.book import move_ids, prefix_hashes

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "2026-10"
RESULTS = ROOT / "lab" / "results"
GROUPS = {"<=900": (600, 900), "1100-1200": (1100, 1200), "1500": (1500, 1500), "1800": (1800, 1800),
          "2100-2200": (2100, 2200), "2300+": (2300, 2800)}
N_BOOK, N_EVAL, CAP, EARLY = 350_000, 10_000, 40, 12
SEED = 0


def load(split: str, n: int, rng) -> dict[str, pl.DataFrame]:
    d = pl.concat([pl.read_parquet(f, columns=["band", "moves"]) for f in sorted((CORPUS / split).glob("*.parquet"))])
    out = {}
    for name, (lo, hi) in GROUPS.items():
        g = d.filter(pl.col("band").is_between(lo, hi) & (pl.col("moves").list.len() >= 10))
        out[name] = g[np.sort(rng.choice(g.height, min(n, g.height), replace=False))]
        print(f"{split} {name}: {out[name].height:,} games", flush=True)
    return out


def keys_of(df: pl.DataFrame):
    ids, off = move_ids(df["moves"])
    return prefix_hashes(ids, off, cap=CAP)


def main():
    rng = np.random.default_rng(SEED)
    books = {}
    for name, df in load("train", N_BOOK, rng).items():
        k, _ = keys_of(df)
        books[name] = np.unique(k, return_counts=True)
    evals = {name: keys_of(df) for name, df in load("heldout", N_EVAL, rng).items()}
    res = {"N_book": N_BOOK, "N_eval": N_EVAL, "groups": list(GROUPS), "cells": []}
    for en, (keys, cnt) in evals.items():
        off = np.zeros(len(cnt) + 1, np.int64)
        np.cumsum(cnt, out=off[1:])
        start = off[:-1]
        pos = np.arange(len(keys)) - np.repeat(start, cnt)
        big = np.iinfo(np.int64).max
        for bn, (bk, bc) in books.items():
            i = np.searchsorted(bk, keys).clip(max=len(bk) - 1)
            m = np.where(bk[i] == keys, bc[i], 0).astype(float)
            first = lambda mask: np.where((f := np.minimum.reduceat(np.where(mask, pos, big), start)) == big, cnt, f) + 1
            par = np.empty(len(keys))
            par[1:] = m[:-1]
            par[start] = N_BOOK
            ok = par >= 100
            weird = (ok & (m < 0.01 * par)).sum() / ok.sum()
            early = (pos < EARLY) & (par > 0)
            logq = np.log10(np.maximum(m[early], 0.5) / par[early])   # unseen child counts as half a game
            cell = {"eval": en, "book": bn, "t1": float(first(m < 1).mean()), "t10": float(first(m < 10).mean()),
                    "weird": float(weird), "logq_early": float(logq.mean())}
            res["cells"].append(cell)
    tab = pl.DataFrame(res["cells"])
    with pl.Config(tbl_cols=10, tbl_width_chars=200, float_precision=3):
        for metric in ("t1", "weird", "logq_early"):
            print(metric, "(rows: eval group, columns: book group)")
            print(tab.pivot(on="book", index="eval", values=metric))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bandbook.json").write_text(json.dumps(res, indent=1) + "\n")



def bench_features(suite: str = "lichess"):
    """Per game and side: mean log10 q of plies 1-20 under each group's book (move-only features) -> bench."""
    from doppel_lab.bench import suite_dir
    BENCH = suite_dir(suite)
    rng = np.random.default_rng(SEED)
    books = {}
    for name, df in load("train", N_BOOK, rng).items():
        k, _ = keys_of(df)
        books[name] = np.unique(k, return_counts=True)
    g = pl.read_parquet(BENCH / "games.parquet", columns=["game_id", "moves"])
    ids, off0 = move_ids(g["moves"])
    keys, cnt = prefix_hashes(ids, off0, cap=20)
    off = np.zeros(len(cnt) + 1, np.int64)
    np.cumsum(cnt, out=off[1:])
    start = off[:-1]
    pos = np.arange(len(keys)) - np.repeat(start, cnt)
    white = pos % 2 == 0
    cols = {"game_id": g["game_id"]}
    for bn, (bk, bc) in books.items():
        i = np.searchsorted(bk, keys).clip(max=len(bk) - 1)
        m = np.where(bk[i] == keys, bc[i], 0).astype(float)
        par = np.empty(len(keys))
        par[1:] = m[:-1]
        par[start] = N_BOOK
        logq = np.where(par > 0, np.log10(np.maximum(m, 0.5) / np.maximum(par, 1)), np.nan)
        for side, msk in (("w", white), ("b", ~white)):
            v = np.where(msk & np.isfinite(logq), logq, 0.0)
            n = np.add.reduceat((msk & np.isfinite(logq)).astype(float), start)
            cols[f"{side}_bb_{bn}"] = np.add.reduceat(v, start) / np.maximum(n, 1)
    f = pl.DataFrame(cols)
    names = list(books)
    for side in ("w", "b"):
        f = f.with_columns(pl.concat_list([pl.col(f"{side}_bb_{n}") for n in names]).list.arg_max()
                           .cast(pl.Float64).alias(f"{side}_bb_argmax"))
    surf = pl.read_parquet(BENCH / "surface.parquet")
    surf = surf.drop([c for c in surf.columns if "_bb_" in c]).join(f, on="game_id")
    surf.write_parquet(BENCH / "surface.parquet")
    print(f"added {f.width - 1} band-book columns -> {BENCH / 'surface.parquet'}", flush=True)


if __name__ == "__main__":
    import sys
    main() if len(sys.argv) < 2 else {"bench": bench_features}[sys.argv[1]](*(sys.argv[2:3] or ["lichess"]))
