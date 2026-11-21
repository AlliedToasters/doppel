"""Shared chess vocabulary for neuron detectors (doppel_lab.detectors), distilled from the 8 pilot detectors
(lab/results/detectors/). Every function is a readable chess concept of one (position, square) or one position, in
the side-to-move frame Maia-3 uses ("us" = side to move, ranks counted from our back rank).

    from doppel_lab.chess_concepts import Pos, gelu
    def detect(board, square, info):
        p = Pos.of(board, info)                 # per-position cache: computed once, reused for all 64 squares
        s = 2.0 * p.guarded_only_by(square, chess.BISHOP, by_us=False) - 1.5 * p.enemy_pawn_guards(square)
        return gelu(s, k=0.5, c=1.5)

Pos caches attack maps, Maia's move-target mass and the Stockfish PV per board, so detectors stay fast (the harness
calls detect ~380k times). Add new concepts here when a detector needs one that others could reuse: a short
docstring in chess terms, side-to-move frame, no engine calls.
"""

from __future__ import annotations

import math

import chess

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))


def gelu(s: float, k: float = 1.0, c: float = 0.0) -> float:
    """Map a pre-activation-like score through GELU(k (s - c)), as the neuron does. Use it when ranking matters:
    clearly-off tokens land near 0 and near-misses at GELU's dip (~ -0.17). For detection alone (AUROC / F1) return
    the raw score instead; GELU's dip reorders near-misses below clear negatives."""
    z = k * (s - c)
    return 0.5 * z * (1.0 + math.erf(z / math.sqrt(2.0)))


class Pos:
    """One board in the side-to-move frame, with per-position caches. Build with Pos.of(board, info)."""

    _last: tuple | None = None

    @classmethod
    def of(cls, board: chess.Board, info: dict | None = None) -> "Pos":
        key = (board.fen(), id(info))
        if cls._last is not None and cls._last[0] == key:
            return cls._last[1]
        p = cls(board, info or {})
        cls._last = (key, p)
        return p

    def __init__(self, board: chess.Board, info: dict):
        self.b = board
        self.info = info
        self.us = board.turn
        self.them = not board.turn
        self._att: dict = {}
        self._to_mass = None
        self._legal_to = None

    # -------------------------------------------------- frame
    def rel_rank(self, sq: int) -> int:
        """Rank counted from our back rank: 0 = our first rank, 7 = theirs."""
        r = chess.square_rank(sq)
        return r if self.us == chess.WHITE else 7 - r

    def rel_square(self, sq: int) -> int:
        """The square as Maia-3 sees it (board mirrored when black is to move)."""
        return sq if self.us == chess.WHITE else chess.square_mirror(sq)

    def edge_distance(self, sq: int) -> int:
        """0 on the a/h files, 1 on b/g, 2 on c/f, 3 on d/e."""
        f = chess.square_file(sq)
        return min(f, 7 - f)

    def center_distance(self, sq: int) -> int:
        """Chebyshev distance to the four centre squares (0 on d4/e4/d5/e5)."""
        f, r = chess.square_file(sq), chess.square_rank(sq)
        return max(min(abs(f - 3), abs(f - 4)), min(abs(r - 3), abs(r - 4)))

    # -------------------------------------------------- occupancy
    def piece(self, sq: int):
        return self.b.piece_at(sq)

    def ours(self, sq: int, piece_type: int | None = None) -> bool:
        p = self.b.piece_at(sq)
        return p is not None and p.color == self.us and (piece_type is None or p.piece_type == piece_type)

    def theirs(self, sq: int, piece_type: int | None = None) -> bool:
        p = self.b.piece_at(sq)
        return p is not None and p.color == self.them and (piece_type is None or p.piece_type == piece_type)

    def empty(self, sq: int) -> bool:
        return self.b.piece_at(sq) is None

    # -------------------------------------------------- control
    def attackers(self, sq: int, by_us: bool) -> chess.SquareSet:
        k = (sq, by_us)
        if k not in self._att:
            self._att[k] = self.b.attackers(self.us if by_us else self.them, sq)
        return self._att[k]

    def guard_types(self, sq: int, by_us: bool) -> set[int]:
        """Piece types of one side that attack the square."""
        return {self.b.piece_type_at(a) for a in self.attackers(sq, by_us)}

    def n_guards(self, sq: int, by_us: bool) -> int:
        return len(self.attackers(sq, by_us))

    def guarded_only_by(self, sq: int, piece_type: int, by_us: bool) -> bool:
        """The square is attacked by that side, and only by pieces of this type (e.g. 'only their bishop sees it')."""
        return self.guard_types(sq, by_us) == {piece_type}

    def guarded_by(self, sq: int, piece_type: int, by_us: bool) -> bool:
        return piece_type in self.guard_types(sq, by_us)

    def cheapest_guard(self, sq: int, by_us: bool) -> int:
        """Value of the cheapest attacker of that side (0 if none): 1 = a pawn guards it, 100 = only the king."""
        return min((VALUE[t] for t in self.guard_types(sq, by_us)), default=0)

    def enemy_pawn_guards(self, sq: int) -> bool:
        return self.guarded_by(sq, chess.PAWN, by_us=False)

    def hanging(self, sq: int) -> bool:
        """A piece (not a king) attacked by the other side and defended by none of its own."""
        p = self.b.piece_at(sq)
        if p is None or p.piece_type == chess.KING:
            return False
        own = p.color == self.us
        return self.n_guards(sq, by_us=not own) > 0 and self.n_guards(sq, by_us=own) == 0

    def safe_for_us(self, sq: int) -> bool:
        """No enemy attacker, or we guard it at least as often and no enemy pawn hits it."""
        n_them = self.n_guards(sq, by_us=False)
        return n_them == 0 or (not self.enemy_pawn_guards(sq) and self.n_guards(sq, by_us=True) >= n_them)

    def contact(self, sq: int) -> int:
        """For the piece on sq: how many squares it attacks that the other side guards (where the armies touch)."""
        p = self.b.piece_at(sq)
        if p is None:
            return 0
        other = self.them if p.color == self.us else self.us
        return sum(1 for t in self.b.attacks(sq) if self.b.is_attacked_by(other, t))

    # -------------------------------------------------- lines and geometry
    def line_hit(self, sq: int, piece_type: int, by_us: bool, dirs: str | None = None) -> int:
        """2 if a piece of this type and side sees sq on a clear line, 1 if through exactly one piece of its own
        colour (x-ray / battery / discovered attack), 0 otherwise. Lines by default: bishop diagonals, rook files and
        ranks, others both; dirs="diag" or "orth" restricts them (e.g. 'on our queen's diagonal', 'king on the
        same diagonal': geometry only, not a legal attack)."""
        color = self.us if by_us else self.them
        if dirs is None:
            dirs = "diag" if piece_type == chess.BISHOP else "orth" if piece_type == chess.ROOK else "both"
        dirs = DIAG if dirs == "diag" else ORTH if dirs == "orth" else DIAG + ORTH
        best = 0
        f0, r0 = chess.square_file(sq), chess.square_rank(sq)
        for df, dr in dirs:
            f, r, blockers = f0 + df, r0 + dr, []
            while 0 <= f < 8 and 0 <= r < 8 and len(blockers) < 2:
                p = self.b.piece_at(chess.square(f, r))
                if p is not None:
                    if p.piece_type == piece_type and p.color == color:
                        if not blockers:
                            return 2
                        if blockers[0] == color:
                            best = 1
                        break
                    blockers.append(p.color)
                f += df
                r += dr
        return best

    def on_line_to(self, sq: int, target: int, max_blockers: int = 1) -> bool:
        """sq lies on the straight line between an enemy slider and our `target` square (king or queen), with at most
        max_blockers pieces between them: pin / skewer / discovered-check geometry."""
        for x in self.b.pieces(chess.BISHOP, self.them) | self.b.pieces(chess.ROOK, self.them) | \
                self.b.pieces(chess.QUEEN, self.them):
            if not chess.ray(x, target):
                continue
            pt = self.b.piece_type_at(x)
            straight = chess.square_file(x) == chess.square_file(target) or \
                chess.square_rank(x) == chess.square_rank(target)
            if (pt == chess.BISHOP and straight) or (pt == chess.ROOK and not straight):
                continue
            btw = chess.between(x, target)
            if chess.popcount(btw & self.b.occupied) <= max_blockers and \
                    (sq in (x, target) or btw & chess.BB_SQUARES[sq]):
                return True
        return False

    def king_distance(self, sq: int, ours: bool = True) -> int:
        k = self.b.king(self.us if ours else self.them)
        return 9 if k is None else chess.square_distance(sq, k)

    def piece_distance(self, sq: int, piece_type: int, ours: bool = True) -> int:
        """Distance to the nearest piece of this type and side (9 if none)."""
        ps = self.b.pieces(piece_type, self.us if ours else self.them)
        return min((chess.square_distance(sq, x) for x in ps), default=9)

    # -------------------------------------------------- pawn structure (side-to-move frame)
    def _file_pawns(self, sq: int, ours: bool) -> list[int]:
        f = chess.square_file(sq)
        return [self.rel_rank(x) for x in self.b.pieces(chess.PAWN, self.us if ours else self.them)
                if chess.square_file(x) == f]

    def own_pawn_ahead(self, sq: int) -> bool:
        """One of our pawns stands further up this file (sq is behind it: a rook's lane or a pawn's support)."""
        return any(r > self.rel_rank(sq) for r in self._file_pawns(sq, True))

    def own_pawn_behind(self, sq: int) -> bool:
        return any(r < self.rel_rank(sq) for r in self._file_pawns(sq, True))

    def file_open(self, sq: int) -> str:
        """'open' (no pawns), 'half_open' (no pawn of ours, theirs present), 'their_half_open', or 'closed'."""
        ours, theirs = bool(self._file_pawns(sq, True)), bool(self._file_pawns(sq, False))
        return "open" if not ours and not theirs else "half_open" if not ours else \
            "their_half_open" if not theirs else "closed"

    def blockade_square(self, sq: int) -> bool:
        """The square directly in front of an enemy pawn (where it would advance)."""
        return any(r == self.rel_rank(sq) + 1 for r in self._file_pawns(sq, False))

    def pawn_push_square(self, sq: int) -> bool:
        """An empty square one of our pawns can step onto (single push)."""
        if not self.empty(sq):
            return False
        back = sq - 8 if self.us == chess.WHITE else sq + 8
        return 0 <= back < 64 and self.ours(back, chess.PAWN)

    def passed(self, sq: int) -> bool:
        """The piece on sq is a passed pawn (no enemy pawn ahead on its own or adjacent files)."""
        p = self.b.piece_at(sq)
        if p is None or p.piece_type != chess.PAWN:
            return False
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = range(r + 1, 8) if p.color == chess.WHITE else range(0, r)
        for ff in (f - 1, f, f + 1):
            if 0 <= ff < 8:
                for rr in ahead:
                    q = self.b.piece_at(chess.square(ff, rr))
                    if q is not None and q.piece_type == chess.PAWN and q.color != p.color:
                        return False
        return True

    def developed(self, sq: int) -> bool:
        """A knight or bishop that has left its own back rank."""
        p = self.b.piece_at(sq)
        if p is None or p.piece_type not in (chess.KNIGHT, chess.BISHOP):
            return False
        r = chess.square_rank(sq)
        return r != (0 if p.color == chess.WHITE else 7)

    # -------------------------------------------------- moves and targets
    def legal_to(self) -> dict[int, list[chess.Move]]:
        """Our legal moves grouped by destination square."""
        if self._legal_to is None:
            d: dict[int, list] = {}
            for m in self.b.legal_moves:
                d.setdefault(m.to_square, []).append(m)
            self._legal_to = d
        return self._legal_to

    def reachable(self, sq: int, piece_type: int | None = None) -> bool:
        """We have a legal move to sq (optionally with a piece of this type)."""
        ms = self.legal_to().get(sq, [])
        return any(piece_type is None or self.b.piece_type_at(m.from_square) == piece_type for m in ms)

    def maia_mass_to(self, sq: int) -> float:
        """Maia-3's probability (top 5 moves) of a move landing on sq."""
        if self._to_mass is None:
            d: dict[int, float] = {}
            for u, p in zip(self.info.get("maia_top5") or [], self.info.get("maia_p5") or []):
                m = chess.Move.from_uci(u)
                d[m.to_square] = d.get(m.to_square, 0.0) + p
            self._to_mass = d
        return self._to_mass.get(sq, 0.0)

    def maia_mass_from(self, sq: int) -> float:
        """Maia-3's probability (top 5) of moving the piece on sq."""
        return sum(p for u, p in zip(self.info.get("maia_top5") or [], self.info.get("maia_p5") or [])
                   if chess.Move.from_uci(u).from_square == sq)

    def in_pv(self, sq: int, plies: int = 2) -> bool:
        """sq is a from- or to-square of the first `plies` moves of Stockfish's principal variation."""
        for u in (self.info.get("sf_pv") or [])[:plies]:
            m = chess.Move.from_uci(u)
            if sq in (m.from_square, m.to_square):
                return True
        return False

    # -------------------------------------------------- position context
    def band(self) -> int:
        """Rating band of the game (700 / 1100 / 1500 / 1900 / 2300)."""
        return int(self.info.get("band") or 1500)

    def band_z(self) -> float:
        """Rating in units of 400 points from 1500 (-2 at 700, +2 at 2300)."""
        return (self.band() - 1500) / 400

    def ply(self) -> int:
        return int(self.info.get("ply") or 0)

    def n_pieces(self) -> int:
        return len(self.b.piece_map())

    def endgame(self) -> float:
        """0 with all 32 men on the board, 1 with 16 or fewer."""
        return min(max((32 - self.n_pieces()) / 16, 0.0), 1.0)

    def in_check(self) -> bool:
        return self.b.is_check()

    def eval_us(self) -> float:
        """Stockfish depth-12 score for the side to move, in pawns, clipped to [-10, 10]."""
        return max(min((self.info.get("sf_cp") or 0) / 100, 10.0), -10.0)

    def obvious(self) -> float:
        """Maia-3's top-move probability: near 1 when there is one obvious move."""
        p = self.info.get("maia_p5") or [0.0]
        return float(p[0])
