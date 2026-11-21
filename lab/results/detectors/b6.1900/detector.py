"""Detector for Maia-3 neuron b6.1900.

Main trigger: empty squares on our back rank in a rook's lane (between our rook and another of our pieces on the
first rank: rooks not yet connected), and the mirror case of empty squares between the enemy king and rook on their
back rank. Secondary: contested squares (attacked by both sides), especially next to our king and occupied ones.
"""
import chess
from doppel_lab.chess_concepts import Pos

# how strongly a rook's lane counts, by the piece of the rook's own side that closes it off
W = {chess.QUEEN: 1.0, chess.BISHOP: 1.0, chess.ROOK: 1.0, chess.KING: 0.6, chess.KNIGHT: 0.4, chess.PAWN: 0.2}


def rank_neighbours(board: chess.Board, sq: int):
    """Nearest occupied square to the left and to the right of sq on its rank (None if empty to the edge)."""
    out = []
    for df in (-1, 1):
        f, r = chess.square_file(sq) + df, chess.square_rank(sq)
        hit = None
        while 0 <= f < 8:
            if board.piece_at(chess.square(f, r)):
                hit = chess.square(f, r)
                break
            f += df
        out.append(hit)
    return out


def rook_lane(p: Pos, sq: int) -> tuple[float, int | None, float]:
    """For an empty square seen by a rook along its rank: (ours, our rook's square, theirs).
    ours is graded by which friendly piece closes the lane on the far side (Q/B/R 1.0, K 0.6, N 0.4, P 0.2,
    open to the edge 0.4, enemy piece 0.1): the rooks are not connected yet. theirs = 1 when the square lies
    between the enemy rook and the enemy king on a rank (their unconnected, uncastled back rank)."""
    if not p.empty(sq):
        return 0.0, None, 0.0
    ours, rook, theirs = 0.0, None, 0.0
    L, R = rank_neighbours(p.b, sq)
    for a, b in ((L, R), (R, L)):
        if a is None or p.b.piece_type_at(a) != chess.ROOK:
            continue
        bp = p.b.piece_at(b) if b is not None else None
        if p.ours(a):
            v = 0.4 if bp is None else W[bp.piece_type] if bp.color == p.us else 0.1
            if v > ours:
                ours, rook = v, a
        elif bp is not None and bp.color == p.them and bp.piece_type == chess.KING:
            theirs = 1.0
    return ours, rook, theirs


def detect(board, square, info):
    p = Pos.of(board, info)
    e = p.empty(square)
    au, at = min(p.n_guards(square, True), 3), min(p.n_guards(square, False), 3)
    lu, rook, lt = rook_lane(p, square)
    s = 0.0
    if rook is not None:
        # our rook's lane on the back rank; strongest when the rook is still in its corner, close by, with the square
        # boxed in from the front, and the square is not part of the move being played
        fwd = square + (8 if p.us == chess.WHITE else -8)
        s += 3.0 * lu + 1.0 * (chess.square_file(rook) in (0, 7) and p.rel_rank(rook) == 0)
        s -= 0.35 * chess.square_distance(rook, square)
        s += 0.9 * (0 <= fwd < 64 and not p.empty(fwd)) - 0.8 * p.in_pv(square, 2) + 0.6 * (at > 0)
        s *= 0.85
    if lt:
        # between the enemy king and rook: next to their king, and not covered by us
        s += 2.4 - 1.6 * min(au, 2) + 1.1 * (p.king_distance(square, ours=False) == 1)
    r0e = e and p.rel_rank(square) == 0
    s += 1.45 * r0e - 0.4 * r0e * au
    # contested squares: attacked by both sides, more so when a piece stands there
    s += 0.55 * (au + at) + 0.55 * min(au, at) + 0.85 * (not e) * (au > 0) * (at > 0)
    kadj = p.king_distance(square) == 1
    s += 0.9 * kadj + 0.2 * kadj * (at > 0) - 0.65 * (p.king_distance(square, ours=False) == 1)
    s += 0.65 * (p.ours(square, chess.PAWN) or p.theirs(square, chess.PAWN))
    s += 0.35 * p.guarded_by(square, chess.KNIGHT, by_us=True) - 0.7 * p.guarded_by(square, chess.PAWN, by_us=True)
    s += 0.8 * min(p.ply(), 80) / 40
    return s
