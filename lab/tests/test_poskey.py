"""poskey (lab/poskey, Rust) keys equal python-chess chess.polyglot.zobrist_hash.

Build first: cargo build --release --manifest-path lab/poskey/Cargo.toml
"""

import random
import subprocess
from pathlib import Path

import chess
import chess.polyglot
import pytest

BIN = Path(__file__).resolve().parents[1] / "poskey/target/release/poskey"
pytestmark = pytest.mark.skipif(not BIN.exists(), reason="poskey not built")

# en passant pinned (b5xc6 would expose the a5 king), castling rights after rook moves, promotions, ep both sides
FENS = [
    ("7k/2p5/8/KP5r/8/8/8/8 b - - 0 1", "c7c5"),
    ("4k3/8/8/8/1p6/8/P1P5/4K3 w - - 0 1", "a2a4 b4a3 c2c4"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "a1b1 h8h7 b1a1 h7h8"),
    ("8/P6k/8/8/8/8/p6K/8 w - - 0 1", "a7a8q a2a1n a8a1"),
    ("rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 3", "d4e3"),
]


def keys(lines: list[str], ep: str = "pseudo") -> list[list[int]]:
    out = subprocess.run([str(BIN), "keys", "--ep", ep], input="\n".join(lines) + "\n", capture_output=True,
                         text=True, check=True).stdout
    return [[int(h, 16) for h in ln.split()] for ln in out.splitlines()]


def reference(fen: str, moves: list[str]) -> list[int]:
    b = chess.Board(fen) if fen else chess.Board()
    hs = [chess.polyglot.zobrist_hash(b)]
    for u in moves:
        b.push_uci(u)
        hs.append(chess.polyglot.zobrist_hash(b))
    return hs


def random_games(n: int, seed: int = 0) -> list[list[str]]:
    rng = random.Random(seed)
    games = []
    for _ in range(n):
        b, mv = chess.Board(), []
        for _ in range(rng.randint(1, 160)):
            legal = list(b.legal_moves)
            if not legal:
                break
            m = rng.choice(legal)
            mv.append(m.uci())
            b.push(m)
        games.append(mv)
    return games


def test_polyglot_random_positions():
    games = random_games(3000)
    got = keys(["|" + " ".join(g) for g in games])
    n = 0
    for g, k in zip(games, got, strict=True):
        assert k == reference("", g)
        n += len(k)
    assert n > 100_000


def test_polyglot_crafted():
    got = keys([f"{f}|{m}" for f, m in FENS])
    for (f, m), k in zip(FENS, got, strict=True):
        assert k == reference(f, m.split()), f


def test_known_start_key():
    assert keys(["|e2e4"])[0] == [0x463B96181691FC9C, 0x823C9B50FD114196]


def test_pinned_ep_is_hashed():
    """Polyglot (and python-chess) hash the ep file when a pawn is adjacent, even if the capture is illegal."""
    pseudo, legal = keys([f"{FENS[0][0]}|{FENS[0][1]}"])[0], keys([f"{FENS[0][0]}|{FENS[0][1]}"], "legal")[0]
    assert pseudo[-1] != legal[-1]
