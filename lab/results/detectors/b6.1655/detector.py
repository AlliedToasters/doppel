"""Detector for Maia-3 neuron b6.1655.

A high-rating neuron (fires ~18x more often in the 2300 band than at 700). Within a position it marks enemy
pieces -- above all knights and bishops -- the further they have advanced into our half, plus squares sitting in
the queens' crossfire or reached only by our queen/king. It is suppressed on squares the enemy guards with a pawn,
on squares our pawns cover, on our own rooks, and on the two back ranks. Output is a log-odds style score.
"""
import chess

V = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 20}
BAND = {700: -2.9, 1100: -2.8, 1500: -2.3, 1900: -1.5, 2300: 0.0}


def _on_enemy_line(board, square, them):
    """Square lies on an enemy bishop/rook/queen line aimed at our king or queen (at most one blocker)."""
    us = not them
    targets = [board.king(us)] + list(board.pieces(chess.QUEEN, us))
    for x in board.pieces(chess.BISHOP, them) | board.pieces(chess.ROOK, them) | board.pieces(chess.QUEEN, them):
        pt = board.piece_type_at(x)
        for t in targets:
            if not chess.ray(x, t):
                continue
            straight = chess.square_file(x) == chess.square_file(t) or chess.square_rank(x) == chess.square_rank(t)
            if (pt == chess.BISHOP and straight) or (pt == chess.ROOK and not straight):
                continue
            btw = chess.between(x, t)
            if chess.popcount(btw & board.occupied) <= 1 and (square in (x, t) or btw & chess.BB_SQUARES[square]):
                return True
    return False


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    us = board.turn
    them = not us
    z = BAND.get(info.get("band"), -2.0)
    rr = chess.square_rank(square) if us == chess.WHITE else 7 - chess.square_rank(square)  # 0 = our back rank
    p = board.piece_at(square)
    if p is not None and p.color == them:
        z += 0.6 + 1.9 * (7 - rr) / 7  # enemy piece, more so the deeper it has come
        if p.piece_type in (chess.KNIGHT, chess.BISHOP):
            z += 1.7
        elif p.piece_type == chess.PAWN:
            z += 0.75
        if chess.square_distance(square, board.king(them)) == 1:  # a king's bodyguard, not an intruder
            z -= 0.6
    elif p is not None and p.piece_type == chess.ROOK:
        z -= 1.1
    if rr in (0, 7):
        z -= 0.9
    au = [V[board.piece_type_at(s)] for s in board.attackers(us, square)]
    at = [V[board.piece_type_at(s)] for s in board.attackers(them, square)]
    if au and min(au) >= 9:  # only our queen / king touch it
        z += 1.0
    elif au and min(au) == 1:
        z -= 0.7
    if at and min(at) == 1:  # enemy pawn guards it
        z -= 1.6
    if 9 in at:  # enemy queen eyes it
        z += 0.75
    if _on_enemy_line(board, square, them):
        z += 0.8
    z += 0.55 * (32 - len(board.piece_map())) / 16
    return z
