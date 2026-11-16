"""Earlier games of the Allie 2022 test-set players, from the Lichess monthly dumps (CPU + network, ~20 min/month).

    python -m doppel_lab.allie_history run                 # every month in MONTHS, resumable, sequential
    python -m doppel_lab.allie_history month 2022-01       # one month (curl | zstd | filter)

Needs data/allie/players.parquet (allie_players.py). Streams each month's standard rated dump, keeps games where
White or Black is a test player and that pass corpus.keep (rated blitz, no BOT, Normal / Time forfeit, both Elo
600-2800, every move clocked: the Doppel's training distribution), and writes data/allie/history/<month>.parquet
with the corpus columns plus utc_ms (game start, from UTCDate + UTCTime). Nothing raw touches the disk. Months run
one at a time, so at most one download is open against database.lichess.org. Whether a history game is "before"
a test game is decided later, per test game, against players.parquet's created_at; the test games themselves are
in the dumps too and are dropped by game_id at that step.

2022 months also write data/allie/base/<month>.parquet: a rating-stratified sample of everyone else's games (no
test player on either side), the base-policy corpus matched to the test set. The test set is flat at ~11% per
200-Elo bucket from 800 to 2200; a month's games pile up at 1400-1800, so buckets are kept at different rates
(BASE_RATE, decided by a hash of the game id, so reruns pick the same games) and flattened exactly later.
Per-bucket seen / kept counts go to data/allie/base/<month>.json.
"""

from __future__ import annotations

import argparse
import calendar
import multiprocessing as mp
import subprocess
import sys
import time
import zlib
import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from doppel_lab.corpus import keep
from doppel_lab.ingest import HDR, parse_movetext

ROOT = Path(__file__).resolve().parents[2]
PLAYERS = ROOT / "data/allie/players.parquet"
OUT = ROOT / "data/allie/history"
BASE = ROOT / "data/allie/base"
BASE_RATE = {600: 1, 800: 1 / 4, 1000: 1 / 8, 1200: 1 / 16, 1400: 1 / 16, 1600: 1 / 16, 1800: 1 / 16, 2000: 1 / 8,
             2200: 1 / 4, 2400: 1, 2600: 1, 2800: 1}   # by 200-Elo bucket of the two players' mean
MONTHS = [f"2021-{m:02d}" for m in (11, 12)] + [f"2022-{m:02d}" for m in range(1, 13)]
UA = "doppel-research-client"
COLS = ("game_id", "white", "black", "white_elo", "black_elo", "band", "base", "inc", "result", "termination",
        "moves", "clocks", "utc_ms")
CHUNK = 2_000


def utc_ms(h: dict) -> int | None:
    try:
        y, mo, d = (int(x) for x in h["UTCDate"].split("."))
        hh, mi, ss = (int(x) for x in h["UTCTime"].split(":"))
        return calendar.timegm((y, mo, d, hh, mi, ss)) * 1000
    except (KeyError, ValueError):
        return None


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
                    h["Result"], h["Termination"], p[0], p[1], utc_ms(h)))
    return out


def games(stream, players: frozenset, base: bool, seen: dict, kept: dict):
    """Yields (is_history, headers, movetext). Header lines are kept raw and only parsed for games with a test
    player, or (base) for rated blitz games whose hash falls under their bucket's rate; most of a month is skipped
    with a few startswith checks per header line."""
    raw: list[str] = []
    hit = blitz = False
    site = we = be = None
    for line in stream:
        if line.startswith("["):
            raw.append(line)
            if line.startswith('[White "') or line.startswith('[Black "'):
                if line[8:-3].lower() in players:
                    hit = True
            elif base:
                if line == '[Event "Rated Blitz game"]\n':
                    blitz = True
                elif line.startswith('[Site "'):
                    site = line[7:-3].rsplit("/", 1)[-1]
                elif line.startswith('[WhiteElo "'):
                    we = line[11:-3]
                elif line.startswith('[BlackElo "'):
                    be = line[11:-3]
        elif line.strip() and raw:
            take = hit
            if not hit and blitz and site and we and be and we.isdigit() and be.isdigit():
                b = min(max((int(we) + int(be)) // 2 // 200 * 200, 600), 2800)
                seen[b] = seen.get(b, 0) + 1
                take = zlib.crc32(site.encode()) / 2 ** 32 < BASE_RATE[b]
            if take:
                headers = {}
                for r in raw:
                    m = HDR.match(r.rstrip("\n"))
                    if m:
                        headers[m[1]] = m[2]
                if keep(headers):
                    if not hit:
                        kept[b] = kept.get(b, 0) + 1
                    yield hit, headers, line
            raw, hit, blitz, site, we, be = [], False, False, None, None, None


def parse_tagged(chunk):
    return [(hist, r) for hist, rows in ((True, parse_chunk([g for h_, *g in chunk if h_])),
                                         (False, parse_chunk([g for h_, *g in chunk if not h_]))) for r in rows]


def write(rows, out):
    tbl = pa.table({c: [r[i] for r in rows] for i, c in enumerate(COLS)})
    pq.write_table(tbl, out.with_suffix(".tmp"), compression="zstd")
    out.with_suffix(".tmp").rename(out)


def month(tag: str, workers: int = 16, source: str | None = None):
    out, bout = OUT / f"{tag}.parquet", BASE / f"{tag}.parquet"
    base = tag.startswith("2022")
    if out.exists() and (bout.exists() or not base):
        print(f"{tag}: exists, skipping", flush=True)
        return
    if out.exists():
        sys.exit(f"{tag}: history exists but base does not; delete {out} to redo the month in one pass")
    OUT.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    pp = pl.read_parquet(PLAYERS)
    players = frozenset(pl.concat([pp["white"], pp["black"]]).drop_nulls().str.to_lowercase().to_list())
    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{tag}.pgn.zst"
    cmd = source or f"curl -sfL -A '{UA}' {url} | zstd -dc --long=31"
    t0 = time.time()
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                            bufsize=1 << 20)
    hist, brows, chunk, n = [], [], [], 0
    seen, kept = {}, {}

    def take(res):
        for h_, r in res:
            (hist if h_ else brows).append(r)

    with mp.get_context("fork").Pool(workers) as pool:
        pend = []
        for g in games(proc.stdout, players, base, seen, kept):
            chunk.append(g)
            if len(chunk) == CHUNK:
                pend.append(pool.apply_async(parse_tagged, (chunk,)))
                chunk = []
                n += CHUNK
                if len(pend) > 4 * workers:
                    take(pend.pop(0).get())
                if n % 200_000 == 0:
                    print(f"{tag}: {n:,} matching games ({len(hist):,} history, {len(brows):,} base so far), "
                          f"{time.time() - t0:.0f}s", flush=True)
        if chunk:
            pend.append(pool.apply_async(parse_tagged, (chunk,)))
        for p in pend:
            take(p.get())
    rc = proc.wait()
    if rc != 0 and source is None:
        sys.exit(f"{tag}: download pipeline exited {rc}; nothing written (re-run to retry)")
    write(hist, out)
    if base:
        write(brows, bout)
        (BASE / f"{tag}.json").write_text(json.dumps({"seen": seen, "kept_after_filter": kept,
                                                      "rate": BASE_RATE}, indent=1) + "\n")
    print(f"{tag}: wrote {len(hist):,} history + {len(brows):,} base games in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "month"])
    ap.add_argument("tag", nargs="?")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--source", help="shell command producing the PGN instead of the download (smoke tests)")
    a = ap.parse_args()
    for tag in (MONTHS if a.cmd == "run" else [a.tag]):
        month(tag, a.workers, a.source)


if __name__ == "__main__":
    main()
