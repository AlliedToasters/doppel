"""Detector for Maia-3 neuron b5.1419.

Reading: the 'contact zone' between the side to move's active pieces and the enemy's pawn/king cover.
  * our developed knight / bishop / queen whose attack set reaches squares guarded by enemy pawns or king
    (more such squares -> stronger), in a busy middlegame, not while we are in check;
  * empty squares guarded by an enemy pawn/king that sit next to our queen (or are eyed by our bishop/queen/king);
  * our king when our queen stands next to it;
  * the opponent's queen when it is advanced and hits our men / our pawn-cover squares; their rook and king
    in thinned-out positions when they come close to our king.
The score is roughly a firing probability; the silent bulk is ordered by typical activation per piece class.
"""
import chess

P, N, B, R, Q, K = chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
# (piece_type, is_side_to_move) -> (base, per-contact-square slope)
W = {(N, True): (0.16, 0.08), (B, True): (0.10, 0.05), (Q, True): (0.12, 0.04), (K, True): (0.22, 0.03),
     (Q, False): (0.12, 0.08)}
# empty, enemy-pawn/king-guarded square: fire rate by king-distance to our queen (9 = no queen)
DQ_EMPTY = {1: 0.48, 2: 0.15, 3: 0.06, 4: 0.03, 9: 0.035}
# typical (median) activation of the silent bulk per class, used only to order non-firing tokens
SILENT = {None: -0.09, P: -0.006, N: -0.075, B: -0.09, R: -0.045, Q: -0.05, K: -0.10}
LAMBDA = 0.4

_cache = {}


def _guard(board, color):
    key = (board.fen(), color)
    g = _cache.get(key)
    if g is None:
        if len(_cache) > 64:
            _cache.clear()
        g = chess.SquareSet()
        for s in board.pieces(P, color) | board.pieces(K, color):
            g |= board.attacks(s)
        _cache[key] = g
    return g


def _fire(board, square, p, info):
    t = board.turn
    oq = board.pieces(Q, t)
    dq = min((chess.square_distance(square, s) for s in oq), default=9)
    if p is None:
        if square not in _guard(board, not t):
            return 0.045 if dq == 1 else 0.002
        v = DQ_EMPTY.get(dq, 0.01)
        at = board.attackers(t, square)
        if any(board.piece_type_at(a) == B for a in at):
            v += 0.08
        if dq > 1 and any(board.piece_type_at(a) == Q for a in at):
            v += 0.08
        if any(board.piece_type_at(a) == K for a in at):
            v += 0.1
        return v
    ours = p.color == t
    n = len(board.piece_map())
    att = board.attacks(square)
    contact = len(att & _guard(board, not p.color))
    if not ours and p.piece_type in (R, K):
        # their rook/king closing in on our king once the board has thinned out
        d = chess.square_distance(square, board.king(t))
        v = 0.04 * min(contact, 5) + 0.05 * max(0, 5 - d)
        return v * max(0.0, (34 - n) / 20)
    w = W.get((p.piece_type, ours))
    if w is None:
        return 0.001
    v = w[0] + w[1] * min(contact, 5)
    rr = chess.square_rank(square) if p.color == chess.WHITE else 7 - chess.square_rank(square)
    if p.piece_type in (N, B, Q):
        if rr == 0:
            v *= 0.3                       # undeveloped piece on its home rank
        if ours and board.attackers(not t, square):
            v *= 0.6                       # our piece is itself under attack
        if not ours and any(board.color_at(s) == t for s in att):
            v += 0.08                      # their queen is hitting one of our men
        if ours and board.is_check():
            v *= 0.6
    if ours and p.piece_type == K and dq == 1:
        v += 0.15
    ply = info.get("ply") or 0
    v *= 0.6 + 0.4 * min(n, 32) / 32       # busy board ...
    v *= 0.6 + 0.4 * min(ply, 30) / 30     # ... past the opening
    return v


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    p = board.piece_at(square)
    s = _fire(board, square, p, info)
    base = SILENT[p.piece_type if p else None]
    if p is None:   # empty squares the enemy covers heavily sit lowest
        base -= 0.01 * min(len(board.attackers(not board.turn, square)), 4)
    return s + LAMBDA * base
