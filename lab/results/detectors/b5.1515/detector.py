"""Detector for Maia-3 neuron b5.1515: an active enemy piece (not a pawn) -- developed off its home square, mobile,
in contact with our men -- strongest for knights and in simplified / later positions."""
import chess
from doppel_lab.chess_concepts import Pos

N, B, R, Q, K = chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
HOME = {N: (1, 6), B: (2, 5), R: (0, 7), Q: (3,), K: (4,)}


def on_home_square(p: Pos, sq: int) -> bool:
    """The piece on sq stands on one of its side's original starting squares (undeveloped)."""
    pc = p.piece(sq)
    if pc is None or pc.piece_type == chess.PAWN:
        return False
    back = 0 if pc.color == chess.WHITE else 7
    return chess.square_rank(sq) == back and chess.square_file(sq) in HOME[pc.piece_type]


def n_officers(p: Pos, ours: bool) -> int:
    """Knights, bishops, rooks and queens one side still has (7 at the start)."""
    b = p.b
    return chess.popcount(b.occupied_co[p.us if ours else p.them] & ~b.pawns & ~b.kings)


def detect(board, square, info):
    p = Pos.of(board, info)
    pc = p.piece(square)
    if pc is None or pc.color == p.us or pc.piece_type == chess.PAWN:
        return -5.0
    pl = min(p.ply(), 80) / 40
    s = {N: 1.3, B: 0.45, R: 0.2, Q: -0.25, K: -1.2}[pc.piece_type]
    s -= (3.1 - 2.8 * pl) * on_home_square(p, square)     # undeveloped piece: silent in the opening only
    s -= 0.3 * (p.rel_rank(square) == 7)
    s += 0.17 * min(p.piece_mobility(square), 14)
    s += 0.56 * (p.threats_made(square) > 0 or p.n_guards(square, by_us=True) > 0)
    s -= 2.5 * n_officers(p, ours=False) / 7               # fewer enemy pieces -> each one lights up more
    s += 0.14 * p.n_guards(square, by_us=False)
    s += 0.15 * p.king_distance(square, ours=False)
    s -= 0.7 * (on_home_square(p, square) and board.has_castling_rights(p.them))  # opponent still uncastled
    return s
