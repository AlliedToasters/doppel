"""Detector for Maia-3 neuron b7.1015. Return a float that is high where the neuron fires.

Reading: "a square in our own camp (our first three ranks) that belongs to our *short-range* men -- our king stands
there, a pawn or knight of ours stands there, a pawn can step onto it, or our knight covers it -- rather than to our
bishops/rooks/queen, and that the enemy does not hit. Much stronger in low-rated games and in quiet positions
(not in check, no single forced move, more pieces traded off)."

Score is a log-odds sum of hand-rounded terms (fitted on dev by logistic regression, then rounded).
"""
import chess


def _rel_rank(square, us):
    return chess.square_rank(square) if us == chess.WHITE else 7 - chess.square_rank(square)


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    us = board.turn
    rr = _rel_rank(square, us)

    s = -1.1
    if square == board.king(us):
        pass                            # our king's square fires on any rank
    elif rr == 3:                       # just outside our camp: rarely fires
        s -= 1.8
    elif rr > 3:                        # enemy half: almost never
        s -= 3.0
    elif rr == 1:                       # our second rank is the core
        s += 0.4

    p = board.piece_at(square)
    if p is None:
        back = square - 8 if us == chess.WHITE else square + 8
        if 0 <= back < 64 and board.piece_at(back) == chess.Piece(chess.PAWN, us):
            s += 1.4                    # a pawn of ours can step here
        if any(True for _ in board.generate_legal_moves(to_mask=chess.BB_SQUARES[square])):
            s += 0.8                    # some piece of ours can move here
    elif p.color != us:
        s -= 0.65
    elif p.piece_type == chess.KING:
        s += 2.65
    elif p.piece_type in (chess.PAWN, chess.KNIGHT):
        s += 1.5
    else:                               # our bishop / rook / queen sits here
        s -= 1.1

    ours = board.attackers(us, square)
    s += 1.5 * len(ours & board.pieces(chess.KNIGHT, us))       # knight-covered square
    s -= 0.8 * len(ours & (board.pieces(chess.ROOK, us) | board.pieces(chess.QUEEN, us)))
    if board.king(us) in ours:          # right next to our king: suppressed
        s -= 1.0
    s -= 0.4 * min(len(board.attackers(not us, square)), 3)     # enemy hits the square

    band = info.get("band") or 1500
    s -= 0.43 * (band - 1500) / 400.0   # lower rating -> fires more
    if board.is_check():
        s -= 1.1
    p1 = (info.get("maia_p5") or [0.4])[0]
    s -= 0.9 * p1                       # one obvious move -> fires less
    s -= 1.85 * len(board.piece_map()) / 32.0                   # crowded board -> fires less
    return s
