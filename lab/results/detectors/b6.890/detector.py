"""Detector for Maia-3 neuron b6.890. Return a float that is high where the neuron fires.

board: chess.Board in real orientation (board.turn = side to move); square: a chess.Square (real coordinates);
info: maia_top5 / maia_p5 (Maia-3's top moves, UCI, and probabilities), played (UCI), sf_cp (Stockfish depth-12
score in centipawns for the side to move), sf_pv (Stockfish principal variation, UCI), ply, band (rating band).
Called once per (position, square). Keep it fast: no engine calls here; use python-chess and info only.

Rule (side-to-move frame): an EMPTY square that our rooks could reach from our own back rank without passing
one of our pawns (no own pawn below it on its file: an open / half-open file, or the squares behind our pawn),
that is covered by our knights, bishops or queen, central rather than on the rim, and not on our back rank.
Bonus on the square directly in front of an enemy pawn (its blockade square, e.g. a6 vs a7 on a half-open
a-file) and when the square is hit by an enemy pawn.  Own pawns, rooks, queen and king on the square switch it off.
"""
import chess

MINOR_Q = (chess.KNIGHT, chess.BISHOP, chess.QUEEN)


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    us = board.turn
    them = not us
    f = chess.square_file(square)
    rank = chess.square_rank(square)
    rr = rank if us == chess.WHITE else 7 - rank          # rank counted from our side
    p = board.piece_at(square)

    z = -5.25
    if p is None:
        z += 0.25
    elif p.color == us and p.piece_type not in (chess.KNIGHT, chess.BISHOP):
        z -= 3.4

    # coverage by our pieces
    minorq = pawns = 0
    for s in board.attackers(us, square):
        pt = board.piece_type_at(s)
        if pt in MINOR_Q:
            minorq += 1
        elif pt == chess.PAWN:
            pawns += 1
    z += 0.9 * min(minorq, 3) + 0.35 * pawns
    if board.is_attacked_by(them, square) and any(
            board.piece_type_at(s) == chess.PAWN for s in board.attackers(them, square)):
        z += 0.4

    # file structure: no own pawn between our back rank and this square; enemy pawn further up the file
    clear = True
    nearest_enemy = 99   # relative rank of the closest enemy pawn above this square on the file
    for s in board.pieces(chess.PAWN, us):
        if chess.square_file(s) == f:
            r2 = chess.square_rank(s) if us == chess.WHITE else 7 - chess.square_rank(s)
            if r2 < rr:
                clear = False
    for s in board.pieces(chess.PAWN, them):
        if chess.square_file(s) == f:
            r2 = chess.square_rank(s) if us == chess.WHITE else 7 - chess.square_rank(s)
            if r2 > rr:
                nearest_enemy = min(nearest_enemy, r2)
    if clear:
        z += 1.75
    if nearest_enemy == rr + 1:          # blockade square right in front of an enemy pawn
        z += 1.2

    z -= (0.0, 0.2, 0.6, 1.3)[min(abs(f - 3), abs(f - 4))]   # centre files best, rim files worst
    if rr == 0:
        z -= 1.65
    return z
