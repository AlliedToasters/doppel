"""Detector for Maia-3 neuron b5.656.

Reading: a "quiet, safe square on the far side from our king": the enemy does not attack it (above all not with a
rook or queen), our rooks look along it, it is two-plus squares from our own king, ideally on our 5th-7th rank;
an own bishop (or pawn / queen) on it is welcome, an own knight or an enemy pawn on it is not. The whole map is
much hotter in 700-rated games and when we are in check.
"""
import chess
from doppel_lab.chess_concepts import Pos

P, N, B, R, Q, K = chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
KD = {0: -4.0, 1: -3.5, 2: -0.7, 7: -0.2}                     # distance to our own king
RANK = {1: -0.6, 4: 0.25, 5: 1.1, 6: 0.6, 7: -0.6}           # counted from our back rank
EMPTY = 1.05                                                  # occupant terms (an own rook / king scores 0)
OURS = {N: -1.65, B: 1.7, P: 0.75, Q: 0.75, R: 0.0, K: 0.0}
THEIRS = {P: -1.05, N: 0.4, B: 0.55, R: 0.35, Q: -0.1, K: 0.0}


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    p = Pos.of(board, info)
    sq = square
    n_them = p.n_guards(sq, by_us=False)
    enemy = p.guard_types(sq, by_us=False)
    s = -1.4 * (n_them > 0) - 1.1 * (n_them > 1) - 1.3 * (R in enemy) - 0.85 * (Q in enemy) - 0.25 * (P in enemy)
    s += 0.15 * min(p.n_guards(sq, by_us=True), 3)
    s += 0.55 * p.line_hit(sq, R, by_us=True) + 0.2 * p.line_hit(sq, Q, by_us=True)
    s += KD.get(p.king_distance(sq, True), 0.0)
    s += RANK.get(p.rel_rank(sq), 0.0)
    pc = board.piece_at(sq)
    if pc is None:
        s += EMPTY
    else:
        s += (OURS if pc.color == p.us else THEIRS)[pc.piece_type]
    s += {700: 1.3, 1100: 0.25, 2300: 0.25}.get(p.band(), 0.0) + 0.75 * p.in_check()
    s -= 0.25 * bool(board.pieces(Q, p.them))
    n = p.n_pieces()
    s -= 0.6 * (p.ply() < 10) + 0.4 * (n < 16) + 0.3 * (n >= 28)   # quietest in the opening and thin endgames
    return s
