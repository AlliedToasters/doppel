"""Detector for Maia-3 neuron b6.849.

Rule (estimate of the pre-activation, then passed through GELU):
  * our own bishop's square: strong (0.8 if on relative ranks 2-5, else 0.45), halved when we are in check;
    our other pieces never fire;
  * any other square: a "landing square" score for our long-range pieces and king - +W_LINE for each of our
    bishop / queen / rook / king attacking it, + Maia's probability of a B/R/Q/K move landing there,
    + a little for squares the piece Maia wants to move can reach; minus enemy coverage (esp. pawns),
    distance from our king, knight-reach, and being in check; the enemy king's square never fires;
  * global modulators: fires far more for low-rated players and when the side to move is losing.
"""
import math

import chess

_cache = {}
GELU = True
G_A = 2.5
G_C = 0.15
BAND_SLOPE = 0.00015
SF_W = 0.1
W_LINE = 0.08   # square attacked by our bishop / queen / rook / king
W_PTO = 1.0     # Maia probability of a B/R/Q/K move landing here
W_REACH = 0.2  # Maia probability of moving a B/R/Q/K piece that can land here
W_EN = 0.04     # per enemy attacker
W_ENP = 0.06    # attacked by an enemy pawn
W_KD = 0.015    # per step of distance from our king
_SLIDE = (chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def _prep(board, info):
    key = board.fen()
    if _cache.get('k') == key:
        return _cache['v']
    us = board.turn
    pf, pto = {}, {}
    for u, p in zip(info.get('maia_top5') or [], info.get('maia_p5') or []):
        mv = chess.Move.from_uci(u)
        pt = board.piece_type_at(mv.from_square)
        pf[mv.from_square] = pf.get(mv.from_square, 0) + p
        if pt in _SLIDE:
            pto[mv.to_square] = pto.get(mv.to_square, 0) + p
    reach = {}
    for mv in board.legal_moves:
        if board.piece_type_at(mv.from_square) in _SLIDE:
            reach[mv.to_square] = max(reach.get(mv.to_square, 0), pf.get(mv.from_square, 0))
    att = {pt: chess.SquareSet() for pt in range(1, 7)}
    for s, p in board.piece_map().items():
        if p.color == us:
            att[p.piece_type] |= board.attacks(s)
    v = (pto, reach, att, board.is_check(), board.king(us))
    _cache['k'] = key; _cache['v'] = v
    return v


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    # The rule below estimates the neuron's pre-activation; pass it through GELU like the model does
    # (so clearly-off squares sit near 0 and near-misses at the GELU minimum, as the real activation does).
    s = _score(board, square, info)
    # weaker players: fires much more (about 4x more squares at 700 than at 2300)
    s -= BAND_SLOPE * ((info.get('band') or 1500) - 1100)
    # side to move is losing: fires on more squares
    sf = info.get('sf_cp') or 0
    s += SF_W * min(max(-sf, 0), 1000) / 1000.0
    if not GELU:
        return s
    x = G_A * (s - G_C)
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _score(board, square, info):
    pto, reach, att, check, king = _prep(board, info)
    us = board.turn
    p = board.piece_at(square)
    if p is not None and p.color == us:
        if p.piece_type != chess.BISHOP:
            return -0.1
        rr = chess.square_rank(square) if us else 7 - chess.square_rank(square)
        s = 0.8 if 1 <= rr <= 4 else 0.45
        return s * (0.5 if check else 1.0)
    if p is not None and p.piece_type == chess.KING:
        return -0.2
    s = (W_LINE * ((square in att[chess.BISHOP]) + (square in att[chess.QUEEN])
                   + (square in att[chess.ROOK]) + (square in att[chess.KING]))
         + W_PTO * pto.get(square, 0) + W_REACH * reach.get(square, 0))
    if square in att[chess.KNIGHT]:
        s -= 0.06
    if p is not None and p.piece_type == chess.KNIGHT:
        s -= 0.05
    enemy = board.attackers(not us, square)
    s -= W_EN * len(enemy) + W_ENP * bool(enemy & board.pieces(chess.PAWN, not us))
    s -= W_KD * chess.square_distance(square, king)
    if check:
        s -= 0.05
    return s
