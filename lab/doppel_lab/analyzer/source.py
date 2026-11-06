"""Load one finished game into a plain record.

Record fields: game_id, white, black, white_elo, black_elo, base, inc (seconds), result, termination, moves (UCI),
clocks (mover's clock after each ply, seconds, or None when the source has no clocks), source.
"""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[3]
V0 = ROOT / "data/v0"
LICHESS_ID = re.compile(r"^(?:https?://)?(?:\w+\.)?lichess\.org/([A-Za-z0-9]{8})|^([A-Za-z0-9]{8})(?:[A-Za-z0-9]{4})?$")


class LiveGame(Exception):
    """The game has no result yet. The analyzer never looks at games in progress."""


def from_pgn(text: str, source: str = "pgn") -> dict:
    g = chess.pgn.read_game(io.StringIO(text))
    if g is None:
        raise ValueError("no game in PGN")
    if g.errors:
        raise ValueError(f"bad PGN: {g.errors[0]}")
    h = g.headers
    if h.get("Result", "*") == "*":
        raise LiveGame("game has no result (still in progress?): refusing to analyze it")
    moves, clocks = [], []
    for node in g.mainline():
        moves.append(node.move.uci())
        clocks.append(node.clock())
    if any(c is None for c in clocks):
        clocks = None
    tc = h.get("TimeControl", "-")
    base, inc = (float(x) for x in tc.split("+")) if "+" in tc else (None, 0.0)

    def elo(k):
        v = h.get(k, "")
        return int(v) if v.isdigit() else 1500

    site = h.get("Site", "")
    gid = site.rsplit("/", 1)[-1] if "lichess.org/" in site else h.get("GameId", "game")
    return {"game_id": gid, "white": h.get("White", "?"), "black": h.get("Black", "?"),
            "white_elo": elo("WhiteElo"), "black_elo": elo("BlackElo"), "base": base, "inc": inc,
            "result": h["Result"], "termination": h.get("Termination", ""), "moves": moves, "clocks": clocks,
            "source": source}


def from_lichess(game_id: str) -> dict:
    url = f"https://lichess.org/game/export/{game_id}?clocks=true&evals=false&opening=false"
    req = urllib.request.Request(url, headers={"Accept": "application/x-chess-pgn", "User-Agent": "doppel-analyzer"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode()
    rec = from_pgn(text, source="lichess")
    rec["game_id"] = game_id
    return rec


def from_v0(game_id: str) -> dict | None:
    import polars as pl

    for f in sorted(V0.glob("games_*_*.parquet")):
        d = pl.read_parquet(f).filter(pl.col("game_id") == game_id)
        if d.height:
            r = d.row(0, named=True)
            return {"game_id": game_id, "white": r.get("white") or "White", "black": r.get("black") or "Black",
                    "white_elo": r["white_elo"], "black_elo": r["black_elo"], "base": float(r["base"]),
                    "inc": float(r["inc"]), "result": r["result"], "termination": r["termination"],
                    "moves": list(r["moves"]), "clocks": [float(c) for c in r["clocks"]] if r["clocks"] else None,
                    "source": f.name}
    return None


def load(spec: str) -> dict:
    p = Path(spec)
    if p.suffix == ".pgn" and p.exists():
        return from_pgn(p.read_text(), source=p.name)
    rec = from_v0(spec) if V0.exists() else None
    if rec is not None:
        return rec
    m = LICHESS_ID.match(spec.strip())
    if m:
        return from_lichess(m.group(1) or m.group(2))
    raise ValueError(f"not a PGN file, a v0 game_id or a Lichess game ID: {spec}")


def validate(rec: dict) -> dict:
    """Replay the moves once so bad input fails here, not halfway through a channel."""
    b = chess.Board()
    for u in rec["moves"]:
        b.push_uci(u)
    if rec["clocks"] is not None and len(rec["clocks"]) != len(rec["moves"]):
        rec["clocks"] = None
    return rec
