"""Detector for Maia-3 neuron b7.946: 'the move is settled; this square is not part of it'.

Position term (sets how many squares fire): Maia-3 is confident (low top-5 entropy, little mass outside its top 5),
the side to move has few legal moves, the move is a capture rather than a check, the game is late (many plies, pawns
left but few pieces), castling rights are gone, material is unbalanced and the side to move is clearly worse.
Square term (which squares fire): far from Maia's candidate moves, away from our king and nearer theirs, not hit by
enemy pieces; our pawns and our king are silent, our rooks/queen/knights fire; our back rank lights up, strongly once
our king has left it.
"""
import math
import chess
from doppel_lab.chess_concepts import Pos

P, N, B, R, Q, K = chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
OUR = {P: -2.35, N: 1.2, B: -0.1, R: 1.55, Q: 1.35, K: -2.0}
THEIR = {P: 0.1, N: -0.25, B: -0.3, R: -0.6, Q: 0.0, K: 0.4}
VAL = {P: 1, N: 3, B: 3, R: 5, Q: 9, K: 0}


def maia_entropy(p: Pos) -> float:
    """Entropy (nats) of Maia-3's top-5 move distribution: ~0 when one move is obvious, ~1.6 when spread."""
    return -sum(q * math.log(q) for q in (p.info.get("maia_p5") or []) if q > 0)


def maia_move_distance(p: Pos, sq: int) -> float:
    """Probability-weighted Chebyshev distance from sq to Maia-3's top-5 candidate moves (nearer of from/to square
    of each move): 0 when sq is on the move everyone expects, large when sq is away from all the action."""
    tops = p.info.get("maia_top5") or []
    ps = p.info.get("maia_p5") or []
    if not tops:
        return 3.0
    tot = sum(ps) or 1.0
    d = 0.0
    for u, q in zip(tops, ps):
        m = chess.Move.from_uci(u)
        d += q * min(chess.square_distance(sq, m.from_square), chess.square_distance(sq, m.to_square))
    return d / tot


def material_balance(b: chess.Board, us: bool) -> int:
    """Material of the side to move minus the opponent's, in pawns (P1 N3 B3 R5 Q9)."""
    return sum(VAL[x.piece_type] * (1 if x.color == us else -1) for x in b.piece_map().values())


def position_score(p: Pos) -> float:
    """How 'settled' the position looks (log-odds-like): see module docstring."""
    b = p.b
    ps = p.info.get("maia_p5") or [0.0]
    top = p.info.get("maia_top5") or []
    m = chess.Move.from_uci(top[0]) if top else None
    s = -1.1 * maia_entropy(p) + 2.3 * sum(ps) - 0.95 * (ps[0] - (ps[1] if len(ps) > 1 else 0.0))
    nl = b.legal_moves.count()
    s -= 0.475 * math.log(nl + 1) + 0.08 * nl / 10
    if m is not None and m in b.legal_moves:
        s += 0.46 * b.is_capture(m) - 0.6 * b.gives_check(m)
    s += -0.35 * b.is_check() + 0.5 * min(p.ply(), 120) / 40 + 0.165 * p.band_z() - 0.27 * b.has_castling_rights(p.us)
    n_pawns = len(b.pieces(P, True)) + len(b.pieces(P, False))
    s += 0.45 * n_pawns / 4 - 0.11 * (p.n_pieces() - n_pawns - 2) / 4
    e = p.eval_us()
    s += 0.055 * max(-e, 0) - 0.39 * math.tanh(-e / 3) - 0.034 * max(e, 0)
    s += 0.07 * abs(material_balance(b, p.us))
    return s


def detect(board, square, info):
    p = Pos.of(board, info)
    if not hasattr(p, "_b7946"):
        p._b7946 = position_score(p)
    s = 1.2 * p._b7946
    s += 0.5 * min(maia_move_distance(p, square), 4.0)
    s += 0.1 * p.king_distance(square, ours=True) - 0.15 * p.king_distance(square, ours=False)
    pc = board.piece_at(square)
    if pc is not None:
        s += (OUR if pc.color == p.us else THEIR)[pc.piece_type]
    s += 0.14 * p.n_guards(square, by_us=True) - 0.36 * p.n_guards(square, by_us=False)
    rr = p.rel_rank(square)
    ku = board.king(p.us)
    king_up = ku is not None and p.rel_rank(ku) >= 2
    s += (0.45 + 0.95 * king_up) * (rr == 0) + 0.6 * king_up * (rr == 1)
    return s
