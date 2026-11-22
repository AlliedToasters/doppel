"""Detector for Maia-3 neuron b8.1728 (always-on neuron; we model its level, not on/off).

Two parts:
  rating / context (same for all 64 squares): weaker players -> higher (band table below, ~0.25 per 400
      Elo and steeper at 2300); side to move worse off, more men on the board, castling rights kept and an
      unclear choice of move (Maia entropy) -> slightly higher.
  square: 'our sphere of action' - squares holding our own pieces (minor pieces / queen / pawns most,
      rooks by how free they are, king a little), empty or enemy squares our legal moves reach (incl.
      pawn pushes), squares we guard, and the from/to squares of the moves Maia expects -> up;
      squares the opponent guards and the opponent's rooks / queen / king -> down.
      The square contrast is itself stronger for weaker players.
"""
import chess
from doppel_lab.chess_concepts import Pos

# mean level by rating band (dev means minus the 1500 mean): stronger players -> lower
BAND = {700: 0.47, 1100: 0.17, 1500: 0.0, 1900: -0.19, 2300: -0.59}
OUR = {chess.PAWN: 0.82, chess.KNIGHT: 1.0, chess.BISHOP: 1.0, chess.QUEEN: 1.0, chess.KING: 0.37, chess.ROOK: 0.06}
THEIR_HEAVY = (chess.ROOK, chess.QUEEN, chess.KING)
_START = chess.Board()


def on_home_square(board: chess.Board, sq: int) -> bool:
    """The man on sq stands on its own starting square from the initial position (unmoved-looking:
    an undeveloped knight/bishop, an unpushed pawn, a rook still in its corner)."""
    pc = board.piece_at(sq)
    return pc is not None and _START.piece_at(sq) == pc


def opening_weight(ply: int, span: int = 40) -> float:
    """1 at the start of the game, falling linearly to 0 by ply `span` (20 moves each)."""
    return max(0.0, 1.0 - ply / span)


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    p = Pos.of(board, info)
    # ---- rating / context part
    ctx = BAND.get(p.band(), -0.245 * p.band_z())
    ctx += -0.04 * max(min(p.eval_us(), 6.0), -6.0) + 0.014 * p.n_pieces()
    ctx += 0.12 * p.maia_entropy() + 0.11 * board.has_castling_rights(board.turn)
    # ---- square part: our pieces and our reach
    s = 0.0
    pc = board.piece_at(square)
    n_to = len(p.legal_to().get(square, []))
    if pc is not None and pc.color == board.turn:
        s += OUR[pc.piece_type]
        if pc.piece_type == chess.ROOK:
            s += 0.7 * min(p.piece_mobility(square), 4) / 4      # a free rook counts like a minor piece
        # opening: men still on their home squares (undeveloped minors, unpushed pawns) up, developed minors down
        op = opening_weight(p.ply())
        home = on_home_square(board, square)
        if pc.piece_type in (chess.KNIGHT, chess.BISHOP):
            s += op * (1.0 if home else -0.6)
        elif pc.piece_type == chess.PAWN:
            s += 0.5 * op * home
        elif pc.piece_type == chess.ROOK:
            s += home * (0.38 - 1.0 * op)
    else:
        if pc is not None:
            s -= 0.17 * (pc.piece_type in THEIR_HEAVY)
            s += 0.15 * (n_to > 0)                              # an enemy man we can capture
        s += 0.59 * (n_to > 0) + 0.19 * (n_to >= 2)              # our legal moves land here
        s += 0.34 * p.pawn_push_square(square)
    s += 0.10 * min(p.n_guards(square, True), 3)
    s -= 0.12 * min(p.n_guards(square, False), 3)
    s += 1.4 * p.maia_mass_to(square) + 1.6 * p.maia_mass_from(square)
    # weaker players show the square pattern more sharply (x1.15 at 700, x0.85 at 2300)
    return ctx + s * (1.0 - 0.075 * p.band_z())
