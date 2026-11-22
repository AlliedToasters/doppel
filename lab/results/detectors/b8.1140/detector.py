"""Detector for Maia-3 neuron b8.1140.

Two parts: (1) a position-wide level set by rating (weaker players -> higher), early opening, and calm (not winning,
few legal moves, no single obvious move) positions, and (2) a
per-square 'in play for us' term: our own minor pieces and pawns, squares we can legally move to, enemy minor
pieces/pawns, and above all the squares of Maia's expected moves; minus our king/rooks, enemy rooks/queen,
squares the opponent piles onto and squares deep in the enemy camp.
"""
import math
import chess
from doppel_lab.chess_concepts import Pos

OWN = {chess.PAWN: 0.3, chess.KNIGHT: 0.35, chess.BISHOP: 0.4, chess.ROOK: 0.0, chess.QUEEN: 0.2, chess.KING: -0.25}
THEIRS = {chess.PAWN: 0.15, chess.KNIGHT: 0.25, chess.BISHOP: 0.25, chess.ROOK: -0.25, chess.QUEEN: -0.25, chess.KING: -0.1}


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    p = Pos.of(board, info)
    opening = min(max((26 - p.ply()) / 12, 0.0), 1.0)  # 1 up to ~move 7, 0 from move 13
    n_moves = sum(len(v) for v in p.legal_to().values())
    level = -0.2 * p.band_z() + 0.35 * opening - 0.1 * p.obvious() - 0.02 * p.eval_us() - 0.006 * n_moves
    pc = board.piece_at(square)
    s = 0.28 * (p.n_guards(square, True) > 0) + 0.33 * p.reachable(square)
    if pc is not None:
        s += (OWN if pc.color == p.us else THEIRS)[pc.piece_type]
    s -= 0.07 * min(p.n_guards(square, False), 3)
    s -= 0.26 * (p.rel_rank(square) == 0) + 0.26 * (p.king_distance(square) == 1)
    s -= 0.12 * (p.rel_rank(square) >= 5) + 0.12 * (p.rel_rank(square) == 7)
    s += 2.0 * math.sqrt(p.maia_mass_to(square) + p.maia_mass_from(square))
    return level + s
