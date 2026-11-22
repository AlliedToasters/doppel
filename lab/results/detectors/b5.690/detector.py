"""Detector for Maia-3 neuron b5.690: enemy-held territory -- squares the opponent occupies or controls that we
(the side to move) do not contest."""
import chess
from doppel_lab.chess_concepts import Pos


def neighbour_control(p: Pos, sq: int, by_us: bool) -> float:
    """Fraction of the king-step neighbours of sq that this side attacks (how much of the area around sq it holds)."""
    nb = chess.SquareSet(chess.BB_KING_ATTACKS[sq])
    return sum(p.n_guards(x, by_us) > 0 for x in nb) / len(nb)


THEIR_W = {chess.PAWN: 2.0, chess.KNIGHT: 1.0, chess.BISHOP: 1.3, chess.ROOK: 1.9, chess.QUEEN: 1.5, chess.KING: 1.8}
OUR_W = {chess.PAWN: 3.4, chess.KNIGHT: 1.8, chess.BISHOP: 2.2, chess.ROOK: 2.3, chess.QUEEN: 2.5, chess.KING: 1.7}


def weighted_control(p: Pos, sq: int, by_us: bool, w: dict) -> float:
    """Sum over one side's attackers of sq of a per-piece-type weight (how firmly that side controls the square)."""
    return sum(w[p.b.piece_type_at(a)] for a in p.attackers(sq, by_us))


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    p = Pos.of(board, info)
    if p.ours(square):
        return -6.0
    s = weighted_control(p, square, False, THEIR_W) - weighted_control(p, square, True, OUR_W)
    s += 2.1 * p.theirs(square) + 1.4 * p.theirs(square, chess.KNIGHT) + 1.3 * p.theirs(square, chess.QUEEN)
    s += 1.5 * neighbour_control(p, square, False) - 1.1 * neighbour_control(p, square, True)
    s += 0.2 * p.rel_rank(square) + 0.03 * (p.n_pieces() - 24)
    return s
