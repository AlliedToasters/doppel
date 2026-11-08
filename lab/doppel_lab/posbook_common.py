"""Shared pieces for the position-book explorations (posbook_*.py): tables, per-ply keys, counts, buckets, results.

Regimes (docs/SCIENCE.md §3): R0 = position key (Polyglot Zobrist, lab/poskey), R1 = move-order prefix key
(book.py). The R0 table is data/v0/posbook/full_p001 (2026-10 train, 0.1% Bloom FPR), the R1 table is
data/v0/book. Both store keys with count >= 2 over the 22.7 M train games, first 80 plies.

The *decision position* of ply k is the position the mover faces: the start for k = 0, else the position after
ply k-1. Counts here are always of decision positions. For games that are themselves in train, `self_exclude`
removes the game's own visits, so a count is "other train games".

DOPPEL_DATA overrides the data directory, e.g. data in the main checkout while running from a git worktree.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.book import move_ids, prefix_hashes

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("DOPPEL_DATA", ROOT / "data"))
CORPUS = DATA / "corpus/2026-10"
V0 = DATA / "v0"
R0_DIR = V0 / "posbook/full_p001"
R1_DIR = V0 / "book"
ANNOT = V0 / "posbook/annot"
BIN = ROOT / "lab/poskey/target/release/poskey"
RESULTS = ROOT / "lab/results/posbook_explore.json"
N_TRAIN = 22_700_959
THREADS = 16
c = pl.col


def load_table(d: Path):
    return np.load(d / "table_keys.npy"), np.load(d / "table_counts.npy").astype(np.int64)


def lookup(tk: np.ndarray, tc: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Counts for query keys (0 when absent: the table holds count >= 2 only)."""
    if len(q) == 0:
        return np.zeros(0, np.int64)
    i = np.searchsorted(tk, q).clip(max=len(tk) - 1)
    return np.where(tk[i] == q, tc[i], 0)


def annotate(name: str, files: list[Path | str], cap: int = 80, force: bool = False):
    """Per-ply position keys (after each ply) for parquet games, via poskey. Returns (keys, lens). Cached."""
    out = ANNOT / name
    if force or not (out / "keys.npy").exists():
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(BIN), "annotate", "--out", str(out), "--cap", str(cap), "--threads", str(THREADS),
                        *map(str, files)], check=True)
    return np.load(out / "keys.npy", mmap_mode="r"), np.load(out / "lens.npy")


def offsets(lens: np.ndarray) -> np.ndarray:
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    return off


def ply_index(lens: np.ndarray):
    """(game index, ply) per flat entry."""
    off = offsets(lens)
    gi = np.repeat(np.arange(len(lens)), lens)
    return gi, np.arange(len(gi)) - off[gi]


def self_visits(keys: np.ndarray, gi: np.ndarray) -> np.ndarray:
    """How often each entry's key occurs in its own game (repetitions within the first plies)."""
    o = np.lexsort((keys, gi))
    k, g = keys[o], gi[o]
    new = np.r_[True, (k[1:] != k[:-1]) | (g[1:] != g[:-1])]
    run = np.cumsum(new) - 1
    n = np.bincount(run)[run]
    out = np.empty(len(keys), np.int64)
    out[o] = n
    return out


def r0_counts(keys: np.ndarray, lens: np.ndarray, table=None, exclude_self: bool = False) -> np.ndarray:
    """R0 count of the position after each ply (flat, like keys)."""
    tk, tc = table if table is not None else load_table(R0_DIR)
    keys = np.asarray(keys)
    cnt = lookup(tk, tc, keys)
    if exclude_self:
        gi, _ = ply_index(lens)
        n = self_visits(keys, gi)
        cnt = np.where(cnt > 0, cnt, n) - n  # absent => the game's own visits were the only ones
        cnt = np.maximum(cnt, 0)
    return cnt


def r1_keys(moves: pl.Series, cap: int = 80):
    ids, off = move_ids(moves)
    return prefix_hashes(ids, off, cap=cap)


def r1_counts(moves: pl.Series, cap: int = 80, table=None, exclude_self: bool = False):
    """R1 count of the move-order prefix after each ply (flat) and per-game lengths."""
    tk, tc = table if table is not None else load_table(R1_DIR)
    keys, lens = r1_keys(moves, cap)
    cnt = lookup(tk, tc, keys)
    if exclude_self:
        cnt = np.maximum(np.where(cnt > 0, cnt, 1) - 1, 0)
    return cnt, lens


def decision_counts(after: np.ndarray, lens: np.ndarray, n_book: int = N_TRAIN) -> np.ndarray:
    """Count of the position the mover faces at each ply: n_book at ply 0, else the count after ply k-1."""
    dec = np.empty(len(after), np.int64)
    dec[1:] = after[:-1]
    off = offsets(lens)
    dec[off[:-1][lens > 0]] = n_book
    return dec


def bucket(cnt: np.ndarray) -> np.ndarray:
    """0: count 0, 1: count 1, then powers of 4: 2: [2, 4), 3: [4, 16), 4: [16, 64), ... capped at 11."""
    return np.digitize(cnt, BUCKET_EDGES).astype(np.int64)


BUCKET_EDGES = np.array([1, 2, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144])


BUCKET_LABELS = ["0", "1", "2-3", "4-15", "16-63", "64-255", "256-1k", "1k-4k", "4k-16k", "16k-65k", "65k-262k",
                 ">=262k"]
BUCKET_X = [0, 1, 2.45, 7.75, 31, 124, 496, 1985, 7940, 31760, 127000, 1e6]   # geometric mid, for charts


def save_section(name: str, obj) -> None:
    """Read-modify-write one top-level key of lab/results/posbook_explore.json (under a lock: jobs run in parallel)."""
    import fcntl
    with open(V0 / "posbook/explore.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        res[name] = obj
        res["date"] = "2026-11-07"
        # Round-trip through _np, then NaN/inf -> null: strict JSON, so the site build can import it.
        res = _finite(json.loads(json.dumps(res, default=_np)))
        RESULTS.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")


def _finite(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    return x


def _np(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


# --- opening names (lichess-org/chess-openings, CC0; data/raw/chess-openings/{a..e}.tsv) ---------------------------

def opening_keys() -> pl.DataFrame:
    """Polyglot key of each named opening's final position, with ECO, name, family and its ply depth."""
    import chess
    import chess.pgn
    import chess.polyglot
    import io
    rows = []
    for f in sorted((DATA / "raw/chess-openings").glob("?.tsv")):
        for r in pl.read_csv(f, separator="\t").iter_rows(named=True):
            g = chess.pgn.read_game(io.StringIO(r["pgn"]))
            b = g.board()
            n = 0
            for m in g.mainline_moves():
                b.push(m)
                n += 1
            rows.append((np.uint64(chess.polyglot.zobrist_hash(b)), r["eco"], r["name"], r["name"].split(":")[0], n))
    d = pl.DataFrame(rows, schema={"key": pl.UInt64, "eco": pl.String, "name": pl.String, "family": pl.String,
                                   "depth": pl.Int64}, orient="row")
    return d.sort("depth", descending=True).unique("key", keep="first")  # deepest name wins on duplicate keys


def name_games(keys: np.ndarray, lens: np.ndarray, op: pl.DataFrame | None = None, max_ply: int = 40) -> pl.DataFrame:
    """Per game: the named opening of the last named position reached in the first max_ply plies (Lichess rule)."""
    op = op if op is not None else opening_keys()
    ok = op["key"].to_numpy()
    order = np.argsort(ok)
    ok = ok[order]
    gi, ply = ply_index(lens)
    keys = np.asarray(keys)
    m = ply < max_ply
    i = np.searchsorted(ok, keys[m]).clip(max=len(ok) - 1)
    hit = ok[i] == keys[m]
    gsel, psel, idx = gi[m][hit], ply[m][hit], order[i[hit]]
    last = pl.DataFrame({"g": gsel, "ply": psel, "idx": idx}).sort("ply").group_by("g").agg(c("idx").last())
    names = op.with_row_index("idx").select(c("idx").cast(pl.Int64), "eco", "name", "family")
    out = pl.DataFrame({"g": np.arange(len(lens))}).join(last.with_columns(c("idx").cast(pl.Int64)), on="g",
                                                          how="left").join(names, on="idx", how="left")
    return out.sort("g").drop("idx")
