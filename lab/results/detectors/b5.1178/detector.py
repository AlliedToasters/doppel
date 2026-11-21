"""Detector for Maia-3 neuron b5.1178.

Main trigger: the enemy's diagonal lines of pressure. Squares on an enemy bishop's (or queen's) diagonal up to the
first piece it meets, strongest when that diagonal runs into our king / queen / rook / bishop (direct attack, or a
pin / x-ray through one of our pawns). Secondary: our own advanced pieces (esp. king, queen, rook) and squares
Maia-3 wants to move to.
"""
import chess
from doppel_lab.chess_concepts import Pos

B, Q, R, N, P, K = chess.BISHOP, chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.PAWN, chess.KING
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
VALUABLE = {K: 1.0, Q: 1.0, R: 0.8, B: 0.8}
ADV = {P: 0.5, N: 0.6, B: 0.6, R: 1.2, Q: 1.4, K: 2.0}


def enemy_diagonal_pressure(p: Pos) -> dict:
    """{square: score} for the enemy's diagonal lines (bishops weight 1, queens 0.6). Each diagonal ray is walked
    from the slider: empty squares score the base weight, the first man met scores too (ours higher). The line is
    'loaded' (+1) when it ends on our K/Q/R/B, either directly or behind exactly one blocker (pin / x-ray / discovered
    attack); then every square on the segment, blocker included, gets the bonus."""
    if getattr(p, "_edp", None) is not None:
        return p._edp
    b, out = p.b, {}
    for x in b.pieces(B, p.them) | b.pieces(Q, p.them):
        w = 1.0 if b.piece_type_at(x) == B else 0.6
        f0, r0 = chess.square_file(x), chess.square_rank(x)
        for df, dr in DIAG:
            seg, f, r, blocker = [], f0 + df, r0 + dr, None
            loaded = 0.0
            while 0 <= f < 8 and 0 <= r < 8:
                s = chess.square(f, r)
                pc = b.piece_at(s)
                if pc is None:
                    if blocker is None:
                        seg.append((s, w))
                else:
                    if pc.color == p.us and pc.piece_type in VALUABLE:
                        loaded = VALUABLE[pc.piece_type] * (1.0 if blocker is None else 0.8)
                        if blocker is None:
                            seg.append((s, w))
                        break
                    if blocker is not None:
                        break
                    blocker = pc
                    seg.append((s, w * (0.8 if pc.color == p.us else 0.0)))
                f += df
                r += dr
            for s, v in seg:
                out[s] = max(out.get(s, 0.0), v + 1.5 * loaded * w)
    p._edp = out
    return out


def detect(board, square, info):
    p = Pos.of(board, info)
    edp = enemy_diagonal_pressure(p)
    own, their = p.ours(square), p.theirs(square)
    rr = p.rel_rank(square)
    edge = p.edge_distance(square) == 0
    kd = p.king_distance(square, ours=False)
    s = 1.0 * own - 1.5 * their
    d = edp.get(square, 0.0)
    gate = 1.0 if (rr >= 3 or not p.empty(square)) else 0.4
    s += 1.4 * d * gate + 1.2 * d * their
    if p.line_hit(square, B, by_us=True) == 2:
        s += 0.5
    s -= 0.5 * p.guarded_by(square, R, by_us=False)
    if d and p.guarded_by(square, B, by_us=True):
        s += 0.5
    if d and kd <= 2 and rr < 7:
        s += 1.0
    if own:
        pt = board.piece_type_at(square)
        if rr >= 3:
            s += 1.2 * ADV[pt] * (0.4 if edge else 1.0)
        if pt == P and p.center_distance(square) == 0:
            s += 0.8
        s -= 0.6 * p.band_z()
        if pt == K:
            s += 1.0 + 2.0 * p.in_check()
    if p.line_hit(square, K, by_us=False, dirs="diag") == 2:
        s += 0.5
    s += 3.0 * p.maia_mass_to(square)
    s += 0.4 * p.enemy_pawn_guards(square)
    s -= 2.0 * (rr in (0, 7))
    s += 0.2 * rr
    s -= 1.0 * edge
    s -= 0.2 * kd
    s -= 0.8 * p.endgame() + 0.2 * p.band_z()
    return s
