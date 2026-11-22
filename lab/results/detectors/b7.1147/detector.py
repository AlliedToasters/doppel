"""Detector for Maia-3 neuron b7.1147: a contested square we cover at least twice that their queen (or a rook) hits
-- a place to challenge / trade off their heavy piece -- favoured when we can move a like piece there, Maia wants to
move there, and we are ahead in material; killed when their pawn or king guards it or our own piece stands on it."""
import chess
from doppel_lab.chess_concepts import Pos
P, N, B, R, Q, K = chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING


def like_for_like_move(p, sq):
    """We have a legal move to sq with a piece of a type (N/B/R/Q) that already attacks sq from their side: moving it
    there offers a trade of like pieces."""
    tt = p.guard_types(sq, by_us=False) - {P, K}
    return any(p.b.piece_type_at(m.from_square) in tt for m in p.legal_to().get(sq, []))


def detect(board, square, info):
    p = Pos.of(board, info)
    tu = p.guard_types(square, True)
    tt = p.guard_types(square, False)
    if not tu or not tt:                       # not a contested square
        return -5.0 + 2.3 * (Q in tt)
    nu, nt = p.n_guards(square, True), p.n_guards(square, False)
    s = 2.3 * (Q in tt) + 0.8 * (R in tt) - 1.2 * (P in tt) - 1.1 * (K in tt)
    s += 2.3 * (nu >= 2) - 0.5 * min(nt, 3) - 0.7 * (Q in tu)
    s += 0.7 * (bool(tu & {R, Q}) and bool(tt & {R, Q}))      # heavy piece meets heavy piece
    s += 1.1 * like_for_like_move(p, square)
    s += 2.3 * min(p.maia_mass_to(square), 0.6)
    pc = p.piece(square)
    if pc is not None:
        if pc.color == p.us:
            s -= 1.4 - 2.0 * (pc.piece_type in (R, Q) and pc.piece_type in tt)   # our heavy piece faces its twin
        if pc.piece_type in (P, K):
            s -= 0.8
        if pc.color == p.them and pc.piece_type == Q:
            s += 2.0
    s -= 0.6 * (bool(board.pieces(Q, p.us)) and bool(board.pieces(Q, p.them)))
    s += 0.35 * max(-3, min(5, p.material_balance()))
    return s
