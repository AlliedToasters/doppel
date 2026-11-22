"""Detector for Maia-3 neuron b5.406: squares our pawns attack (pawn-controlled squares), strongest when the pawn
is the only guard, when the enemy also contests the square, or an enemy man stands on it (a pawn capture)."""
import chess
from doppel_lab.chess_concepts import Pos


def our_pawn_attacks(p: Pos, sq: int) -> int:
    """How many of our pawns attack sq (0, 1 or 2)."""
    return sum(1 for a in p.attackers(sq, True) if p.b.piece_type_at(a) == chess.PAWN)


def pawn_attackers_blocked_by_own(p: Pos, sq: int) -> int:
    """How many of our pawns attacking sq are stuck behind one of our own men (cannot advance; a 'bad' pawn)."""
    fw = 8 if p.us == chess.WHITE else -8
    return sum(1 for a in p.attackers(sq, True)
               if p.b.piece_type_at(a) == chess.PAWN and 0 <= a + fw < 64 and p.ours(a + fw))


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    p = Pos.of(board, info)
    pa = our_pawn_attacks(p, square)
    if pa == 0:
        return -3.0
    s = 1.0 + 1.0 * (pa == 2)
    others = min(p.n_guards(square, True) - pa, 3)
    nt = min(p.n_guards(square, False), 3)
    s += 0.8 * nt - 0.5 * others + 0.15 * nt * others
    if p.theirs(square):
        s += 0.8
    elif p.ours(square):
        pt = p.b.piece_type_at(square)
        s -= {chess.PAWN: 0.0, chess.KNIGHT: 0.5, chess.BISHOP: 0.4, chess.ROOK: 1.0, chess.QUEEN: 1.0,
              chess.KING: 1.2}[pt]
    s += 0.4 * (p.rel_rank(square) >= 3)
    s += 1.0 * p.maia_mass_to(square) + 0.4 * p.blockade_square(square) + 0.3 * p.pawn_push_square(square)
    s -= 0.5 * p.endgame()
    s -= 0.5 * pawn_attackers_blocked_by_own(p, square)
    return max(s, 0.0) * (1.0 + 0.25 * p.band_z())
