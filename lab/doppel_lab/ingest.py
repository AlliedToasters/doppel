"""Stream a (possibly truncated) Lichess PGN .zst dump into a Parquet of band-filtered blitz games.

Keeps: Event "Rated Blitz game", both players in the same 100-Elo band (band lower bounds below),
no BOT titles, Termination Normal or Time forfeit, every move carrying a [%clk].
Output columns: game_id, white, black, white_elo, black_elo, band, base, inc, result, termination,
moves (UCI list), clocks (seconds remaining after each ply).

Usage: python -m doppel_lab.ingest <in.pgn.zst> <out.parquet> [--max-per-band N]
"""

import argparse
import io
import re
import sys
from collections import Counter

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

BANDS = (1100, 1300, 1500, 1700, 1900, 2100)
HDR = re.compile(r'^\[(\w+) "(.*)"\]$')
CLK = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")
# SAN token followed by its comment: "e4 { [%clk 0:05:00] }"
MOVE = re.compile(r"(?:\d+\.(?:\.\.)?\s*)?([^\s{}]+)\s*\{([^}]*)\}")


def band_of(w: int, b: int) -> int | None:
    for lo in BANDS:
        if lo <= w < lo + 100 and lo <= b < lo + 100:
            return lo
    return None


def keep_headers(h: dict) -> int | None:
    if h.get("Event") != "Rated Blitz game":
        return None
    if h.get("WhiteTitle") == "BOT" or h.get("BlackTitle") == "BOT":
        return None
    if h.get("Termination") not in ("Normal", "Time forfeit"):
        return None
    try:
        return band_of(int(h["WhiteElo"]), int(h["BlackElo"]))
    except (KeyError, ValueError):
        return None


def parse_movetext(text: str):
    board = chess.Board()
    moves, clocks = [], []
    for san, comment in MOVE.findall(text):
        m = CLK.search(comment)
        if m is None:
            return None
        try:
            mv = board.push_san(san)
        except ValueError:
            return None
        moves.append(mv.uci())
        clocks.append(int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]))
    return (moves, clocks) if moves else None


def iter_games(path: str):
    """Yield (headers, movetext) pairs; tolerates a truncated final frame."""
    fh = open(path, "rb")
    reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh, read_across_frames=True)
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    headers: dict = {}
    try:
        for line in text:
            line = line.rstrip("\n")
            if line.startswith("["):
                m = HDR.match(line)
                if m:
                    headers[m[1]] = m[2]
            elif line.strip() and headers:
                yield headers, line
                headers = {}
    except zstandard.ZstdError as e:  # truncated download: stop cleanly
        print(f"stream ended: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--max-per-band", type=int, default=20000)
    args = ap.parse_args()

    cols = {k: [] for k in ("game_id", "white", "black", "white_elo", "black_elo", "band", "base", "inc",
                            "result", "termination", "moves", "clocks")}
    per_band = Counter()
    seen = 0
    for h, movetext in iter_games(args.src):
        seen += 1
        if seen % 500000 == 0:
            print(f"{seen:,} games scanned; kept {dict(per_band)}", file=sys.stderr, flush=True)
        band = keep_headers(h)
        if band is None or per_band[band] >= args.max_per_band:
            continue
        try:
            base, inc = (int(x) for x in h["TimeControl"].split("+"))
        except (KeyError, ValueError):
            continue
        parsed = parse_movetext(movetext)
        if parsed is None:
            continue
        moves, clocks = parsed
        per_band[band] += 1
        cols["game_id"].append(h["Site"].rsplit("/", 1)[-1])
        cols["white"].append(h["White"])
        cols["black"].append(h["Black"])
        cols["white_elo"].append(int(h["WhiteElo"]))
        cols["black_elo"].append(int(h["BlackElo"]))
        cols["band"].append(band)
        cols["base"].append(base)
        cols["inc"].append(inc)
        cols["result"].append(h["Result"])
        cols["termination"].append(h["Termination"])
        cols["moves"].append(moves)
        cols["clocks"].append(clocks)
        if all(per_band[b] >= args.max_per_band for b in BANDS):
            break

    pq.write_table(pa.table(cols), args.out, compression="zstd")
    print(f"done: {seen:,} scanned; kept {dict(sorted(per_band.items()))} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
