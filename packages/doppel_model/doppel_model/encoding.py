"""Board/history encoding and move vocabulary (Maia-3 layout).

- Every position is seen from the side to move: a position with Black to move is mirrored first.
- One position = (64, 12) one-hot planes: P N B R Q K for the mover's colour after mirroring, then the same for the other.
- Input = the last `history` positions concatenated on the feature axis (oldest first, padded by repeating the
  earliest), plus one clk_ponder column (always 0 at inference): shape (64, 12 * history + 1).
- Moves are 4096 from-to pairs (from-major, squares a1=0..h8=63) plus 256 promotions a7..h7 -> a8..h8 x q r b n,
  all in the mover's mirrored frame.
"""

from __future__ import annotations

import chess
import numpy as np

_PIECE = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}

MOVES: list[str] = [chess.square_name(a) + chess.square_name(b) for a in range(64) for b in range(64)] + [
    f"{f}7{t}8{p}" for f in "abcdefgh" for t in "abcdefgh" for p in "qrbn"]
MOVE_INDEX = {m: i for i, m in enumerate(MOVES)}


def mirror_uci(u: str) -> str:
    return u[0] + str(9 - int(u[1])) + u[2] + str(9 - int(u[3])) + u[4:]


def board_planes(board: chess.Board) -> np.ndarray:
    b = board.mirror() if board.turn == chess.BLACK else board
    x = np.zeros((64, 12), np.float32)
    for sq, piece in b.piece_map().items():
        x[sq, _PIECE[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)] = 1.0
    return x


def encode(planes: list[np.ndarray], history: int = 8) -> np.ndarray:
    """planes: board_planes() of every position so far, oldest first (at least one)."""
    h = planes[-history:]
    h = [h[0]] * (history - len(h)) + h
    return np.concatenate(h + [np.zeros((64, 1), np.float32)], axis=1)


def legal_mask(board: chess.Board) -> tuple[np.ndarray, list[str]]:
    """Boolean mask over MOVES and, aligned with its True entries in vocab order, the real-frame UCI moves."""
    white = board.turn == chess.WHITE
    pairs = []
    for m in board.legal_moves:
        u = m.uci()
        pairs.append((MOVE_INDEX[u if white else mirror_uci(u)], u))
    pairs.sort()
    mask = np.zeros(len(MOVES), bool)
    mask[[i for i, _ in pairs]] = True
    return mask, [u for _, u in pairs]
