"""Detector for Maia-3 neuron b5.210. Return a float that is high where the neuron fires.

Concept: "diagonal danger/opportunity squares". The neuron fires on squares (mostly empty, in the middle ranks)
that an ENEMY BISHOP covers and nothing else of the enemy does, plus squares on the diagonals of OUR queen,
and squares on enemy bishop/queen diagonals that are only blocked by one enemy piece (x-ray / discovery).
Other enemy control (pawns, knights, rooks, king, queen) of the square suppresses it. Stronger at higher ratings.

board: chess.Board in real orientation (board.turn = side to move); square: a chess.Square (real coordinates);
info: maia_top5 / maia_p5, played, sf_cp, sf_pv, ply, band (rating band).
"""
import math

import chess

RAW = False
K, C = 0.5, 1.5
DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _diag_hit(board, square, piece_type, color):
    """2 if a `color` `piece_type` sees `square` on a clear diagonal, 1 if through exactly one piece of
    `color` itself (x-ray / battery / discovery), else 0."""
    best = 0
    f0, r0 = chess.square_file(square), chess.square_rank(square)
    for df, dr in DIRS:
        f, r, blockers = f0 + df, r0 + dr, []
        while 0 <= f < 8 and 0 <= r < 8 and len(blockers) < 2:
            p = board.piece_at(chess.square(f, r))
            if p:
                if p.piece_type == piece_type and p.color == color:
                    if not blockers:
                        return 2
                    if blockers[0] == color:
                        best = 1
                    break
                blockers.append(p.color)
            f += df
            r += dr
    return best


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    us = board.turn
    them = not us
    enemy = {board.piece_type_at(x) for x in board.attackers(them, square)}
    ours = {board.piece_type_at(x) for x in board.attackers(us, square)}
    piece = board.piece_at(square)

    s = 0.0
    # 1. who of the enemy guards the square
    if enemy == {chess.BISHOP}:
        s += 2.4            # controlled by an enemy bishop and nothing else
    elif enemy and chess.BISHOP not in enemy:
        s -= 1.7            # guarded only by non-bishop enemy pieces
    if enemy & {chess.ROOK, chess.KING}:
        s -= 0.7
    # 2. latent enemy diagonals: bishop / queen x-raying through one of its own pieces
    if _diag_hit(board, square, chess.BISHOP, them) == 1:
        s += 1.2
    if _diag_hit(board, square, chess.QUEEN, them) == 1:
        s += 0.6
    if piece is not None and piece.color == them and piece.piece_type == chess.BISHOP:
        s += 2.5            # the enemy bishop's own square
    # enemy bishop line that also runs to our king (check / pin geometry)
    if _diag_hit(board, square, chess.BISHOP, them) and _diag_hit(board, square, chess.KING, us) == 2:
        s += 1.2
    # 3. our own pieces that could use the square
    s += {2: 1.7, 1: 0.7}.get(_diag_hit(board, square, chess.QUEEN, us), 0.0)  # on our queen's diagonal
    if chess.KNIGHT in ours:
        s += 0.9            # our knight can jump there
    if chess.BISHOP in ours:
        s -= 1.9            # already covered by our own bishop: neuron stays quiet
    # 4. occupancy, rank, game phase, rating
    if piece is None:
        s += 2.0
    elif piece.color == them:
        s += 0.6
    rel_rank = chess.square_rank(square) if us == chess.WHITE else 7 - chess.square_rank(square)
    if rel_rank in (0, 7):
        s -= 1.5
    s += 0.5 * ((info.get("band") or 1500) - 1500) / 400      # stronger for stronger players
    s -= 0.65 * min(info.get("ply") or 0, 80) / 20             # fades as the game goes on
    if RAW:
        return s
    # The score acts like the neuron's pre-activation; pass it through GELU like the real neuron does
    # (so "clearly off" squares sit near 0 and "almost on" squares sit at GELU's negative dip).
    z = K * (s - C)
    return 0.5 * z * (1.0 + math.erf(z / math.sqrt(2.0)))
