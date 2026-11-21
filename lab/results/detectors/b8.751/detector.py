"""Detector for Maia-3 neuron b8.751. Return a float that is high where the neuron fires.

Concept (side-to-move frame): "rook room behind our wing pawns". The neuron fires on our rooks and on the
empty squares of our camp that a rook of ours reaches and/or that sit behind one of our own pawns on the same
file, most strongly on the a/b and g/h files (classic peak: Ra1, empty a2, pawn a3). Enemy control of the
square lowers it. Score is a logit-like sum of a few readable terms (weights rounded from a logistic fit on dev).
"""
import chess


def _reach(board, color, piece_type, square):
    return any(square in board.attacks(q) for q in board.pieces(piece_type, color))


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    us = board.turn
    p = board.piece_at(square)
    rel = square if us else chess.square_mirror(square)   # side-to-move frame: we move up the board
    r = chess.square_rank(rel); f = chess.square_file(rel)
    edge = min(f, 7 - f)                                   # 0 = a/h file, 1 = b/g file
    file_bb = chess.BB_FILES[f]
    bm = board if us else board.mirror()
    own_pawns = bm.pieces_mask(chess.PAWN, chess.WHITE)
    enemy_pawn_on_file = bool(bm.pieces_mask(chess.PAWN, chess.BLACK) & file_bb)
    ahead = file_bb & ~((1 << (rel + 1)) - 1)              # squares further up the same file
    behind_own_pawn = bool(own_pawns & ahead)
    enemy_att = len(board.attackers(not us, square))

    if p is not None:
        if p.color == us and p.piece_type == chess.ROOK:
            s = -0.5
            s += 0.6 * enemy_pawn_on_file                  # rook facing an enemy pawn on its file
            s += 0.4 * behind_own_pawn                     # rook behind its own pawn
            s += 0.9 * (enemy_att > 0)
            s += {0: 0.6, 1: 0.5}.get(edge, 0.0)
            return s
        if p.color == us and p.piece_type == chess.QUEEN:
            return -2.7
        return -4.0                                        # any other occupied square: silent

    s = -4.35
    s += 0.7 * behind_own_pawn
    if rel + 8 < 64 and bm.piece_at(rel + 8) == chess.Piece(chess.PAWN, chess.WHITE):
        s += 0.65                                          # our pawn directly in front
    s += 1.1 * _reach(board, us, chess.ROOK, square)       # a rook of ours can go there
    s -= 0.5 * _reach(board, us, chess.QUEEN, square)
    s += {0: 1.1, 1: 0.7}.get(edge, 0.0)
    s += 0.35 * (1 <= r <= 3)
    s += 0.3 * enemy_pawn_on_file
    s -= 0.4 * min(enemy_att, 3)
    s += 0.4 * min(len(board.attackers(us, square)), 3)
    return s
