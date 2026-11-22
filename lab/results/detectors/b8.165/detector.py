"""b8.165: a rating/uncertainty prior on every square, plus a 'where my move happens' square map."""
import math
import chess
from doppel_lab.chess_concepts import Pos


def our_pv_square(p: Pos, sq: int, moves: int = 3) -> bool:
    """sq is a from- or to-square of one of OUR first `moves` moves in Stockfish's principal variation."""
    for u in (p.info.get("sf_pv") or [])[0:2 * moves:2]:
        m = chess.Move.from_uci(u)
        if sq in (m.from_square, m.to_square):
            return True
    return False


def detect(board, square, info):
    p = Pos.of(board, info)
    # position level: weaker players, unclear (spread) Maia predictions, fuller boards, and being worse
    # all raise the neuron on every square (band alone gives dev spearman 0.38)
    s = -0.245 * p.band_z() - 0.54 * p.obvious() + 0.014 * p.n_pieces() - 0.03 * p.eval_us()
    # square level: squares on Maia's candidate moves (from or to), our planned PV squares,
    # our pieces, squares we can legally move to, squares we control rather than the opponent
    mt, mf = p.maia_mass_to(square), p.maia_mass_from(square)
    s += 0.73 * (mt + mf > 0) + 1.4 * math.sqrt(mt) + 1.0 * math.sqrt(mf)
    s += 0.25 * our_pv_square(p, square)
    s += 0.42 * p.ours(square) + 0.40 * bool(p.legal_to().get(square))
    s += 0.08 * min(p.n_guards(square, True), 3) - 0.15 * min(p.n_guards(square, False), 3)
    # small corrections: own king square and our back rank are quieter, minor pieces louder
    pt = board.piece_type_at(square)
    s -= 0.5 * (pt == chess.KING and p.ours(square))
    s += 0.25 * (pt in (chess.KNIGHT, chess.BISHOP))
    s -= 0.25 * (p.rel_rank(square) == 0)
    return s
