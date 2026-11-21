"""Detector for Maia-3 neuron b5.64 (see notes.md)."""
import chess
from doppel_lab.chess_concepts import Pos

PT = {None: 0.0, chess.PAWN: 0.9, chess.KNIGHT: 1.9, chess.BISHOP: -0.2, chess.ROOK: 0.8, chess.QUEEN: -0.9, chess.KING: -1.3}
KT = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.5, 2.5, 2.5, 2.5]  # 0.5 per step away from their king, capped at 5
KD = [-0.9, 0.8, 1.3, 0.75, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]


def advanced_enemy_pawn_attacks(p: Pos, max_rel_rank: int = 4) -> chess.SquareSet:
    """Squares hit by enemy pawns that have advanced to rel. rank <= max_rel_rank (their 4th rank or further, i.e.
    pawns pressing into our half). Cached per position."""
    key = ("adv_pawn_att", max_rel_rank)
    c = p.__dict__.setdefault("_extra", {})
    if key not in c:
        s = chess.SquareSet()
        for x in p.b.pieces(chess.PAWN, p.them):
            if p.rel_rank(x) <= max_rel_rank:
                s |= p.b.attacks(x)
        c[key] = s
    return c[key]


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    p = Pos.of(board, info)
    pc = board.piece_at(square)
    s = 0.75 * p.band_z() + 0.25 * min(p.ply() / 40, 2) - 1.6 * p.n_pieces() / 32
    s += KD[p.king_distance(square, True)]
    s += KT[p.king_distance(square, False)]
    if pc is not None and pc.color == p.them:
        s += 1.2 * p.endgame() - 0.6
    s += PT[pc.piece_type if pc else None]
    s += 1.5 * p.maia_mass_from(square) - 0.8 * p.maia_mass_to(square)
    s += 0.1 * p.eval_us()
    s += 1.2 * (square in advanced_enemy_pawn_attacks(p))
    them = p.guard_types(square, by_us=False)
    us = p.guard_types(square, by_us=True)
    s += 2.0 * (chess.PAWN in them) + 0.9 * (chess.BISHOP in them) - 0.5 * (chess.ROOK in them) \
        - 1.0 * (chess.QUEEN in them) - 1.1 * (chess.KING in them) - 0.5 * min(p.n_guards(square, False), 3)
    s += 0.6 * min(p.n_guards(square, True), 2)
    s += 0.3 * (chess.ROOK in us) + 0.8 * (chess.KING in us) - 0.6 * (chess.QUEEN in us) - 0.4 * (chess.BISHOP in us)
    s -= 0.7 * (p.line_hit(square, chess.BISHOP, by_us=False) == 2)
    return s
