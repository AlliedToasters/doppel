"""Detector for Maia-3 neuron b8.738 (see notes.md).

Two parts: a position-wide level (rating band, being lost, Maia unsure, calm board) and a per-square bump on the
squares of Maia's expected moves and squares we control / can move to / move from with active pieces.
"""
import chess
from doppel_lab.chess_concepts import Pos

P, N, B, R, Q, K = chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING


def legal_from_count(p: Pos, sq: int) -> int:
    """Number of our legal moves that start on sq (0 for empty / enemy squares)."""
    if "legal_from" not in p._x:
        d: dict[int, int] = {}
        for m in p.b.legal_moves:
            d[m.from_square] = d.get(m.from_square, 0) + 1
        p._x["legal_from"] = d
    return p._x["legal_from"].get(sq, 0)


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    p = Pos.of(board, info)
    # ---- position level
    if "lvl" not in p._x:
        lvl = 3.0 - 0.2 * p.band_z()                         # weaker players -> stronger activation
        lvl += 0.07 * max(0.0, -p.eval_us() - 4.0) - 0.025 * p.eval_us()   # worse for the side to move, esp. lost
        lvl -= 1.2 * sum(info.get("maia_p5") or [1.0])        # Maia unsure: top-5 covers little mass
        lvl += 0.2 * board.has_castling_rights(p.us)          # opening-ish, king not yet committed
        lvl -= 0.01 * sum(len(v) for v in p.legal_to().values())  # fewer options -> a bit louder
        lvl -= 0.08 * (p.n_hanging(True) + p.n_hanging(False))  # calm board, nothing en prise
        p._x["lvl"] = lvl
    s = p._x["lvl"]
    # ---- square level
    s += 1.6 * p.maia_mass_to(square) ** 0.5 + 1.2 * p.maia_mass_from(square) ** 0.5    # squares of the expected move
    s += 0.09 * p.n_guards(square, True) - 0.16 * p.n_guards(square, False)
    s += 0.2 * len(p.legal_to().get(square, [])) + 0.05 * legal_from_count(p, square)
    s -= 0.2 * p.empty(square)
    s += 0.2 * p.reachable(square, P) - 0.2 * p.reachable(square, K)   # pawn-advance squares up, king steps down
    s += 0.1 * p.safe_for_us(square)
    s += (-0.3, -0.1, 0.0, 0.08, 0.12, 0.15, 0.12, 0.1)[p.rel_rank(square)]  # our half quieter, their half louder
    pc = board.piece_at(square)
    if pc is not None:
        if pc.color == p.us:
            s += {N: 0.25, B: 0.35, R: 0.3, Q: 0.28, K: -0.2}.get(pc.piece_type, 0.0)
        else:
            s -= 0.26 * (pc.piece_type == K)
            s += 0.35 * bool(p.attackers(square, True))       # enemy man we hit
    return s
