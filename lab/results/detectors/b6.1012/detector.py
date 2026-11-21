"""Detector for Maia-3 neuron b6.1012.

Chess reading: "loose or stranded enemy pieces" - an enemy man that is undefended, sits on its own back rank or has
come deep into our half, and has little room to move - set against a strong position context: the neuron is far
more active in low-rated (700-1100) games and when the side to move is losing / short of material, where it also
spreads onto empty squares in the enemy half.
"""
import chess
from doppel_lab.chess_concepts import Pos, VALUE

TYPE_W = {chess.PAWN: 0.0, chess.KNIGHT: 1.7, chess.BISHOP: 1.1, chess.ROOK: 1.4, chess.QUEEN: 1.0, chess.KING: 1.4}


def piece_mobility(p, sq):
    """Squares the piece on sq attacks that are not occupied by its own side (pseudo-mobility, pins ignored)."""
    pc = p.b.piece_at(sq)
    return chess.popcount(p.b.attacks_mask(sq) & ~p.b.occupied_co[pc.color])


def side_material(p, ours=True):
    """Material of one side in pawns (P1 N3 B3 R5 Q9, king not counted)."""
    c = p.us if ours else p.them
    return sum(VALUE[t] * len(p.b.pieces(t, c)) for t in (1, 2, 3, 4, 5))


def still_castleable(p, sq):
    """The enemy king or rook on sq keeps a castling right (has not moved; castling is still a plan)."""
    b = p.b
    if b.piece_type_at(sq) == chess.KING:
        return b.has_castling_rights(b.color_at(sq))
    return bool(b.castling_rights & chess.BB_SQUARES[sq])


def detect(board, square, info):
    p = Pos.of(board, info)
    bz = p.band_z()
    lose = max(-p.eval_us(), 0.0) / 5                     # 0..2: Stockfish says we are losing
    lowmat = max(20 - side_material(p, True), 0) / 10     # 0..2: we have little material left
    s = -0.85 * bz + 0.55 * lowmat - 0.15 * lose * bz
    if p.empty(square):
        return s - 2.2 + 0.8 * (p.king_distance(square, ours=True) == 1) + 0.75 * lose + 0.5 * lowmat + 1.4 * p.rel_rank(square) / 7
    if p.ours(square):
        pt = board.piece_type_at(square)
        if pt == chess.ROOK:
            s += 1.6 * (p.rel_rank(square) == 0) - 0.15 * min(piece_mobility(p, square), 10)
        return s + 0.4 * (pt != chess.KING and p.n_guards(square, by_us=True) == 0)
    pt = board.piece_type_at(square)
    s += 0.7 + TYPE_W[pt] + 0.35 * bz + 0.2 * lose
    if pt != chess.KING and p.n_guards(square, by_us=False) == 0:
        s += 1.5
    s += 0.8 * (p.n_guards(square, by_us=True) > 0)
    rr = p.rel_rank(square)
    s += 1.2 if rr == 7 else 1.1 * max(6 - rr, 0) / 6
    kd = p.king_distance(square, ours=True)
    s += 2.2 if kd == 1 else 0.6 if kd == 2 else 0.0        # an enemy man in contact with our king
    if pt in (chess.KING, chess.ROOK) and still_castleable(p, square):
        s -= 0.8
    if pt != chess.PAWN:
        s -= 0.17 * min(piece_mobility(p, square), 10)    # the less room it has, the stronger
    return s
