"""Detector for Maia-3 neuron b6.1739: squares the opponent attacks (empty, or holding one of our pieces), above
all the square in front of our advanced pawn and our own king when its flight squares are covered; stronger in
low-rated, lopsided, in-check and long, piece-rich games."""
import chess
from doppel_lab.chess_concepts import Pos

P, N, B, R, Q, K = chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING


def king_ring_attacked(p: Pos, sq: int) -> int:
    """Number of squares around sq (king's neighbourhood) attacked by the opponent."""
    return sum(p.n_guards(s, by_us=False) > 0 for s in chess.SquareSet(chess.BB_KING_ATTACKS[sq]))


def our_hanging(p: Pos) -> int:
    """How many of our pieces (side to move) are attacked and undefended: a position with loose material."""
    if not hasattr(p, "_our_hanging"):
        p._our_hanging = sum(p.hanging(s) for s in p.b.piece_map() if p.ours(s))
    return p._our_hanging


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    p = Pos.of(board, info)
    at = p.n_guards(square, by_us=False)
    enemy = p.guard_types(square, by_us=False)
    ours = p.guard_types(square, by_us=True)
    rr = p.rel_rank(square)
    # the opponent's control of the square
    s = 1.15 * (at > 0) + 0.55 * min(at, 3)
    s += 0.25 * bool(enemy & {P, B, Q}) - 0.35 * bool(enemy & {R, K})
    s -= 0.45 * (P in ours) + 0.5 * (N in ours)
    # occupant
    s -= 1.6 * p.ours(square, P) + 1.1 * p.theirs(square)
    s -= 0.9 * (p.ours(square, N) or p.ours(square, B))
    s += 0.5 * p.ours(square, R)
    s += 0.2 * p.edge_distance(square)
    if at and p.reachable(square):
        s += 0.55
    if p.pawn_push_square(square):
        s += 0.9 * max(rr - 2, 0)
    if p.ours(square, K):
        s += 1.1 * king_ring_attacked(p, square) * min(max((p.n_pieces() - 8) / 8, 0.0), 1.0)
    # position context
    s -= 0.6 * p.band_z()
    s += 0.15 * min(abs(p.eval_us()), 5)
    s += 1.15 * p.in_check()
    s += 0.3 * min(our_hanging(p), 2)
    s -= 0.3 * p.b.has_castling_rights(p.us)
    s += 0.02 * min(p.ply(), 150) + 0.07 * p.n_pieces()
    return s
