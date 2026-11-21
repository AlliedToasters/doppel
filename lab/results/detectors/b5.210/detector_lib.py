"""b5.210 rewritten on doppel_lab.chess_concepts: same dev score as detector.py (F1 0.4331, AUROC 0.853)."""
import chess
from doppel_lab.chess_concepts import Pos, gelu
B, Q, N, R, K = chess.BISHOP, chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.KING


def detect(board, square, info):
    p = Pos.of(board, info)
    enemy = p.guard_types(square, by_us=False)
    s = 0.0
    if enemy == {B}:
        s += 2.4
    elif enemy and B not in enemy:
        s -= 1.7
    if enemy & {R, K}:
        s -= 0.7
    s += 1.2 * (p.line_hit(square, B, by_us=False) == 1) + 0.6 * (p.line_hit(square, Q, by_us=False, dirs="diag") == 1)
    s += 2.5 * p.theirs(square, B)
    if p.line_hit(square, B, by_us=False) and p.line_hit(square, K, by_us=True, dirs="diag") == 2:
        s += 1.2
    s += {2: 1.7, 1: 0.7}.get(p.line_hit(square, Q, by_us=True, dirs="diag"), 0.0)
    s += 0.9 * p.guarded_by(square, N, by_us=True) - 1.9 * p.guarded_by(square, B, by_us=True)
    s += 2.0 * p.empty(square) + 0.6 * p.theirs(square)
    s -= 1.5 * (p.rel_rank(square) in (0, 7))
    s += 0.5 * p.band_z() - 0.65 * min(p.ply(), 80) / 20
    return gelu(s, k=0.5, c=1.5)
