"""Detector for Maia-3 neuron b5.633. Return a float that is high where the neuron fires.

Graded rule: "a square the side to move has its hands on, and that is in play":
  + covered by our queen / bishop / rook (pawn cover counts against),
  + the square is our king's square or next to it,
  + it is contested: we cover it with a piece AND the opponent attacks it,
  + it holds an enemy piece (and not one of ours),
  + we have a legal move there, Maia-3 puts probability on moving there, or Stockfish's
    first move / the opponent's best reply lands there,
  + stronger rating band and endgames (few pieces) fire more often.
"""
import chess

W_ATT = {chess.QUEEN: 1.0, chess.BISHOP: 0.6, chess.ROOK: 0.3, chess.KNIGHT: 0.0,
         chess.KING: 0.0, chess.PAWN: -0.3}

_cache = {}


def _dest(board, mv):
    if board.is_castling(mv):  # the rook's landing square counts too
        f = 3 if chess.square_file(mv.to_square) == 2 else 5
        return (mv.to_square, chess.square(f, chess.square_rank(mv.to_square)))
    return (mv.to_square,)


def _pos(board, info):
    key = board.fen()
    if _cache.get("key") != key:
        tg = set()
        for mv in board.legal_moves:
            tg.update(_dest(board, mv))
        p_to = {}
        for mv, p in zip(info.get("maia_top5") or [], info.get("maia_p5") or []):
            try:
                m = chess.Move.from_uci(mv)
            except ValueError:
                continue
            for d in _dest(board, m):
                p_to[d] = p_to.get(d, 0.0) + p
        band = info.get("band") or 1500
        base = 0.5 * min(max((band - 700) / 400, 0), 2)
        if len(board.piece_map()) <= 12:
            base += 0.45
        pv = info.get("sf_pv") or []
        pv_sq = set()
        for mv in pv[:2]:  # engine's best move and the opponent's best reply
            try:
                pv_sq.add(chess.Move.from_uci(mv).to_square)
            except ValueError:
                pass
        _cache.update(key=key, tg=tg, p_to=p_to, base=base, pv=pv_sq)
    return _cache


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    c = _pos(board, info)
    us = board.turn
    s = c["base"]
    piece_cover = False
    for a in board.attackers(us, square):
        t = board.piece_type_at(a)
        s += W_ATT[t]
        piece_cover |= t != chess.PAWN
    if piece_cover and board.attackers(not us, square):
        s += 1.5
    pc = board.piece_at(square)
    if pc is not None:
        s += 0.9 if pc.color != us else -0.45
    k = board.king(us)
    if k is not None and chess.square_distance(k, square) <= 1:
        s += 1.6
    if square in c["tg"]:
        s += 0.6
    s += 1.4 * c["p_to"].get(square, 0.0)
    if square in c["pv"]:
        s += 0.6
    return s
