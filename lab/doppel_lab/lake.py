"""Game lake: any PGN stream -> one broad Parquet schema shared by every source (CPU, parallel).

    curl -sL <lichess month .pgn.zst> | zstd -dc --long=31 | python -m doppel_lab.lake - data/lake/lichess/2013-01 \\
        --source lichess --max-games 200000
    unzip -p twic1600g.zip | python -m doppel_lab.lake - data/lake/twic/1600 --source twic

Unlike corpus.py, nothing is filtered by event, rating, title or termination. Casual games, BOT accounts,
titles, dates and event names are what the population studies contrast. The only games dropped are those
whose movetext doesn't replay (illegal SAN, odd variants).

Movetext may span several lines and carry comments, variations and NAGs (TWIC, broadcasts); only the
main line is kept. Clocks: [%clk] per ply when *every* ply has one, else null. Lichess players get the
split.held_out flag (heldout_white/black) so lake rows can be filtered against the v0 held-out set.
Headers not given their own column are kept as a JSON string in `extra`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
import threading
import time
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from doppel_lab.split import held_out

HDR = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
CLK = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")
COMMENT = re.compile(r"\{([^}]*)\}")
TOKEN = re.compile(r"\{[^}]*\}|\(|\)|;[^\n]*|\$\d+|\d+\.(?:\.\.)?|1-0|0-1|1/2-1/2|\*|[^\s(){};$]+")
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}

STR_COLS = ("event", "site", "date", "round", "white", "black", "white_title", "black_title", "result",
            "termination", "time_control", "eco", "opening", "variant", "fen")
HEADER_OF = {"event": "Event", "site": "Site", "date": "Date", "round": "Round", "white": "White",
             "black": "Black", "white_title": "WhiteTitle", "black_title": "BlackTitle", "result": "Result",
             "termination": "Termination", "time_control": "TimeControl", "eco": "ECO", "opening": "Opening",
             "variant": "Variant", "fen": "FEN"}
INT_COLS = {"white_elo": "WhiteElo", "black_elo": "BlackElo", "white_fide_id": "WhiteFideId",
            "black_fide_id": "BlackFideId"}
OWN = set(HEADER_OF.values()) | set(INT_COLS.values()) | {"UTCDate", "UTCTime"}
SCHEMA = pa.schema([("source", pa.string()), ("game_id", pa.string())]
                   + [(c, pa.string()) for c in STR_COLS]
                   + [(c, pa.int32()) for c in ("white_elo", "black_elo")]
                   + [(c, pa.int64()) for c in ("white_fide_id", "black_fide_id")]
                   + [("utc", pa.string()), ("heldout_white", pa.bool_()), ("heldout_black", pa.bool_()),
                      ("plies", pa.int16()), ("moves", pa.list_(pa.string())), ("clocks", pa.list_(pa.float32())),
                      ("extra", pa.string())])
SHARD = 250_000
CHUNK = 2_000


def parse_movetext(text: str, fen: str | None = None):
    """Main-line UCI moves and per-ply clocks (None unless every ply has one). None if it doesn't replay."""
    try:
        board = chess.Board(fen) if fen else chess.Board()
    except ValueError:
        return None
    moves, clocks, depth, clk_ok = [], [], 0, True
    for tok in TOKEN.findall(text):
        c = tok[0]
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif depth:
            continue
        elif c == "{":
            m = CLK.search(tok)
            if m and moves and len(clocks) == len(moves) - 1:
                clocks.append(int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]))
        elif c in ";$" or tok in RESULTS or tok[-1] == ".":
            continue
        else:
            tok = tok.rstrip("!?")
            if moves and len(clocks) != len(moves):
                clk_ok = False
            try:
                moves.append(board.push_san(tok).uci())
            except ValueError:
                return None
    if not moves:
        return None
    if len(clocks) != len(moves):
        clk_ok = False
    return moves, (clocks if clk_ok else None)


def to_int(s):
    try:
        v = int(s)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def row(source: str, h: dict, text: str):
    if h.get("Variant", "Standard") not in ("Standard", "From Position") and source == "lichess":
        return None
    p = parse_movetext(text, h.get("FEN"))
    if p is None:
        return None
    site = h.get("Site", "")
    if source == "lichess" and "lichess.org/" in site:
        gid = site.rsplit("/", 1)[-1]
    else:  # stable content hash: same game from two TWIC issues dedups by id
        key = "|".join(h.get(k, "") for k in ("Event", "Site", "Date", "Round", "White", "Black")) + "|" + " ".join(p[0])
        gid = hashlib.blake2b(key.encode(), digest_size=8).hexdigest()
    r = {"source": source, "game_id": gid}
    for c, k in HEADER_OF.items():
        r[c] = h.get(k)
    for c, k in INT_COLS.items():
        r[c] = to_int(h.get(k))
    r["utc"] = f"{h['UTCDate']} {h.get('UTCTime', '')}".strip() if "UTCDate" in h else None
    lich = source == "lichess"
    r["heldout_white"] = held_out(h["White"]) if lich and "White" in h else None
    r["heldout_black"] = held_out(h["Black"]) if lich and "Black" in h else None
    r["plies"], r["moves"], r["clocks"] = len(p[0]), p[0], p[1]
    ex = {k: v for k, v in h.items() if k not in OWN}
    r["extra"] = json.dumps(ex, ensure_ascii=False) if ex else None
    return r


def parse_chunk(args):
    source, chunk = args
    out, bad = [], 0
    for h, text in chunk:
        r = row(source, h, text)
        if r is None:
            bad += 1
        else:
            out.append(r)
    return out, bad


def decoded(fh):
    """Lines as text: UTF-8, else Windows-1252 (older TWIC issues), else UTF-8 with replacement."""
    for raw in fh:
        try:
            yield raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                yield raw.decode("cp1252")
            except UnicodeDecodeError:
                yield raw.decode("utf-8", errors="replace")


def games(stream):
    """(headers, movetext) pairs; movetext may span lines. Tolerates junk between games."""
    headers, body = {}, []
    for line in stream:
        s = line.strip()
        if s.startswith("[") and s.endswith("]") and (HDR.match(s)):
            if body and headers:
                yield headers, " ".join(body)
                headers, body = {}, []
            elif body:
                body = []
            m = HDR.match(s)
            headers[m[1]] = m[2]
        elif s and headers:
            body.append(s)
    if body and headers:
        yield headers, " ".join(body)


class Writer:
    def __init__(self, root: Path):
        self.root, self.buf, self.n = root, [], 0
        root.mkdir(parents=True, exist_ok=True)
        self.part = len(list(root.glob("part-*.parquet")))

    def add(self, rows):
        self.buf.extend(rows)
        if len(self.buf) >= SHARD:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        tbl = pa.Table.from_pylist(self.buf, schema=SCHEMA)
        path = self.root / f"part-{self.part:04d}.parquet"
        pq.write_table(tbl, path.with_suffix(".tmp"), compression="zstd")
        path.with_suffix(".tmp").rename(path)
        self.part += 1
        self.n += len(self.buf)
        self.buf = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="PGN path, or - for stdin")
    ap.add_argument("out")
    ap.add_argument("--source", required=True, help="lichess | twic | elite | broadcast | lumbra | fics | ...")
    ap.add_argument("--max-games", type=int, default=0, help="stop after this many games read (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    stream = decoded(sys.stdin.buffer if a.src == "-" else open(a.src, "rb"))
    w = Writer(Path(a.out))
    t0, bad, sent = time.time(), 0, 0
    inflight = threading.BoundedSemaphore(3 * a.workers)
    lock = threading.Lock()

    def done(res):
        nonlocal bad
        with lock:
            w.add(res[0])
            bad += res[1]
        inflight.release()

    def err(e):
        print(f"worker error: {e}", file=sys.stderr, flush=True)
        inflight.release()

    with mp.get_context("fork").Pool(a.workers) as pool:
        chunk = []
        for g in games(stream):
            chunk.append(g)
            sent += 1
            if len(chunk) == CHUNK:
                inflight.acquire()
                pool.apply_async(parse_chunk, ((a.source, chunk),), callback=done, error_callback=err)
                chunk = []
            if a.max_games and sent >= a.max_games:
                break
        if chunk:
            inflight.acquire()
            pool.apply_async(parse_chunk, ((a.source, chunk),), callback=done, error_callback=err)
        pool.close()
        pool.join()
    w.flush()
    print(json.dumps({"out": a.out, "read": sent, "kept": w.n, "dropped": bad,
                      "minutes": round((time.time() - t0) / 60, 2)}), flush=True)


if __name__ == "__main__":
    main()
