"""Detector for Maia-3 neuron b6.212: contact squares -- where the two armies touch.

High on our men that are attacked (more so when advanced, passed, or themselves attacking enemy men), on enemy men we
attack, and on empty squares both sides fight over or that Maia wants to move to; stronger in endgames and at higher
ratings.
"""
import chess
from doppel_lab.chess_concepts import Pos


def threats_made(p: Pos, sq: int) -> int:
    """How many enemy men the piece on sq attacks (0 for an empty square)."""
    pc = p.piece(sq)
    if pc is None:
        return 0
    return sum(1 for x in p.b.attacks(sq) if (q := p.b.piece_at(x)) is not None and q.color != pc.color)


def passer_path(p: Pos, sq: int, max_dist: int = 2) -> int:
    """1-2 if sq is on the file ahead of a passed pawn (either colour) with nothing in between, within max_dist steps
    (2 = the very next square, 1 = two steps on); 0 otherwise. The road a passed pawn wants to run."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    best = 0
    for col, step in ((chess.WHITE, -1), (chess.BLACK, 1)):
        rr, dist = r + step, 1
        while 0 <= rr < 8 and dist <= max_dist:
            x = chess.square(f, rr)
            pc = p.b.piece_at(x)
            if pc is not None:
                if pc.piece_type == chess.PAWN and pc.color == col and p.passed(x):
                    best = max(best, max_dist + 1 - dist)
                break
            rr += step
            dist += 1
    return best


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    p = Pos.of(board, info)
    nu, nt = min(p.n_guards(square, True), 3), min(p.n_guards(square, False), 3)
    ctx = 0.85 * min(p.band_z(), 0.0) + 0.4 * min(p.ply(), 120) / 40
    if p.ours(square):
        # our man in contact: attacked (by how many), attacking enemy men, advanced, passed or blockading a pawn
        s = -4.8 + 0.9 * (nt > 0) + 0.7 * nt + 0.36 * min(threats_made(p, square), 3)
        s += 0.6 * min(max(p.rel_rank(square) - 2, 0), 4) + 0.7 * p.passed(square) + 0.7 * p.blockade_square(square)
        s += 0.5 * p.maia_mass_from(square) + 0.1 * nu - 0.4 * (nt > 0 and nu == 0) + 0.35 * p.endgame()
    elif p.theirs(square):
        # an enemy man we hit
        s = -7.6 + 1.8 * (nu > 0) + 1.0 * nu + 0.6 * p.maia_mass_to(square) + 1.4 * p.endgame() + 0.4 * p.in_pv(square, 2)
    else:
        # an empty square both sides fight over, Maia wants to land on, or a passed pawn's road
        s = -8.4 + 1.0 * (nu > 0 and nt > 0) + 0.6 * min(nu, nt) + 0.4 * nu + 0.13 * nt + 2.3 * p.maia_mass_to(square)
        s += 1.0 * (p.rel_rank(square) >= 6) + 2.1 * p.endgame() + 0.4 * p.in_pv(square, 2)
        s += 0.3 * (p.king_distance(square, False) <= 2) + 1.9 * (passer_path(p, square) > 0)
    return s + ctx
