"""Training corpus: stream a full Lichess month into player-split Parquet shards (CPU, parallel).

    curl -sL https://database.lichess.org/standard/lichess_db_standard_rated_2026-10.pgn.zst \
      | zstd -dc --long=31 | python -m doppel_lab.corpus - data/corpus/2026-10

Nothing raw touches the disk. Keeps rated blitz games (any rating 600-2800, unlike ingest.py's same-band eval
filter), no BOT accounts, Termination Normal or Time forfeit, every move carrying a [%clk].
Split (doppel_lab.split.held_out): a game with any held-out player goes to heldout/, else train/, so no
held-out player is ever trained on. Columns as ingest.py (band = 100-Elo bucket of the mean rating).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import threading
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from doppel_lab.ingest import HDR, parse_movetext
from doppel_lab.split import held_out

COLS = ("game_id", "white", "black", "white_elo", "black_elo", "band", "base", "inc", "result", "termination",
        "moves", "clocks")
SHARD = 500_000
CHUNK = 4_000


def keep(h: dict) -> bool:
    if h.get("Event") != "Rated Blitz game" or "BOT" in (h.get("WhiteTitle"), h.get("BlackTitle")):
        return False
    if h.get("Termination") not in ("Normal", "Time forfeit"):
        return False
    try:
        return 600 <= int(h["WhiteElo"]) <= 2800 and 600 <= int(h["BlackElo"]) <= 2800
    except (KeyError, ValueError):
        return False


def parse_chunk(chunk):
    out = []
    for h, text in chunk:
        if "%clk" not in text:
            continue
        try:
            base, inc = (int(x) for x in h["TimeControl"].split("+"))
        except (KeyError, ValueError):
            continue
        p = parse_movetext(text)
        if p is None:
            continue
        we, be = int(h["WhiteElo"]), int(h["BlackElo"])
        out.append((h["Site"].rsplit("/", 1)[-1], h["White"], h["Black"], we, be, (we + be) // 200 * 100, base, inc,
                    h["Result"], h["Termination"], p[0], p[1]))
    return out


def games(stream):
    headers: dict = {}
    for line in stream:
        if line.startswith("["):
            m = HDR.match(line.rstrip("\n"))
            if m:
                headers[m[1]] = m[2]
        elif line.strip() and headers:
            if keep(headers):
                yield headers, line
            headers = {}


class Writer:
    def __init__(self, root: Path):
        self.root, self.buf, self.n = root, {"train": [], "heldout": []}, {"train": 0, "heldout": 0}
        for k in self.buf:
            (root / k).mkdir(parents=True, exist_ok=True)
        self.part = {k: len(list((root / k).glob("part-*.parquet"))) for k in self.buf}

    def add(self, rows):
        for r in rows:
            k = "heldout" if held_out(r[1]) or held_out(r[2]) else "train"
            self.buf[k].append(r)
            if len(self.buf[k]) >= SHARD:
                self.flush(k)

    def flush(self, k):
        rows = self.buf[k]
        if not rows:
            return
        tbl = pa.table({c: [r[i] for r in rows] for i, c in enumerate(COLS)})
        path = self.root / k / f"part-{self.part[k]:04d}.parquet"
        pq.write_table(tbl, path.with_suffix(".tmp"), compression="zstd")
        path.with_suffix(".tmp").rename(path)
        self.part[k] += 1
        self.n[k] += len(rows)
        self.buf[k] = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="decompressed PGN path, or - for stdin")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    stream = sys.stdin if a.src == "-" else open(a.src, encoding="utf-8", errors="replace")
    w = Writer(Path(a.out))
    t0, kept, sent = time.time(), 0, 0
    inflight = threading.BoundedSemaphore(3 * a.workers)  # backpressure: never buffer the whole month
    lock = threading.Lock()

    def done(rows):
        nonlocal kept
        with lock:
            w.add(rows)
            kept += len(rows)
        inflight.release()

    with mp.get_context("fork").Pool(a.workers) as pool:
        chunk = []
        for g in games(stream):
            chunk.append(g)
            if len(chunk) == CHUNK:
                inflight.acquire()
                pool.apply_async(parse_chunk, (chunk,), callback=done, error_callback=lambda e: (print(e, file=sys.stderr), inflight.release()))
                sent += len(chunk)
                chunk = []
                if sent % 1_000_000 == 0:
                    el = time.time() - t0
                    print(f"{sent:,} candidate games sent, {kept:,} kept, {el / 60:.1f} min, {sent / el:,.0f}/s",
                          file=sys.stderr, flush=True)
        if chunk:
            inflight.acquire()
            pool.apply_async(parse_chunk, (chunk,), callback=done)
        pool.close()
        pool.join()
    with lock:
        for k in w.buf:
            w.flush(k)
    print(f"done: {kept:,} games -> train {w.n['train']:,}, heldout {w.n['heldout']:,} in {(time.time() - t0) / 60:.1f} min",
          file=sys.stderr)


if __name__ == "__main__":
    main()
