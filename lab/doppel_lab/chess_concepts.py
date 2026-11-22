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
_START = chess.Board()
_OFFICERS = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


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
        self._x: dict = {}

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

    # -------------------------------------------------- material
    def side_material(self, ours: bool = True) -> int:
        """Material of one side in pawns (P1 N3 B3 R5 Q9, king not counted); ours=True for the side to move."""
        c = self.us if ours else self.them
        return sum(VALUE[t] * len(self.b.pieces(t, c)) for t in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                                                  chess.ROOK, chess.QUEEN))

    def material_balance(self) -> int:
        """Our material minus theirs, in pawns: + when the side to move is ahead."""
        return self.side_material(True) - self.side_material(False)

    def n_hanging(self, ours: bool = True) -> int:
        """How many of one side's pieces are attacked and undefended (loose material on the board)."""
        k = ("n_hanging", ours)
        if k not in self._x:
            c = self.us if ours else self.them
            self._x[k] = sum(self.hanging(s) for s, pc in self.b.piece_map().items() if pc.color == c)
        return self._x[k]

    # -------------------------------------------------- piece activity
    def piece_mobility(self, sq: int) -> int:
        """Squares the piece on sq attacks that its own side does not occupy (pseudo-mobility, pins ignored)."""
        pc = self.b.piece_at(sq)
        if pc is None:
            return 0
        return chess.popcount(self.b.attacks_mask(sq) & ~self.b.occupied_co[pc.color])

    def threats_made(self, sq: int) -> int:
        """How many men of the other side the piece on sq attacks (0 for an empty square)."""
        pc = self.b.piece_at(sq)
        if pc is None:
            return 0
        return chess.popcount(self.b.attacks_mask(sq) & self.b.occupied_co[not pc.color])

    # -------------------------------------------------- king safety and castling
    def king_ring_attacked(self, ours: bool = True) -> int:
        """Squares next to one side's king that the other side attacks (pressure on the king's neighbourhood)."""
        k = ("ring", ours)
        if k not in self._x:
            ksq = self.b.king(self.us if ours else self.them)
            self._x[k] = 0 if ksq is None else sum(
                self.n_guards(s, by_us=not ours) > 0 for s in chess.SquareSet(chess.BB_KING_ATTACKS[ksq]))
        return self._x[k]

    def still_castleable(self, sq: int) -> bool:
        """The king or rook on sq still holds a castling right (it has not moved; castling is still a plan)."""
        pt = self.b.piece_type_at(sq)
        if pt == chess.KING:
            return self.b.has_castling_rights(self.b.color_at(sq))
        return pt == chess.ROOK and bool(self.b.castling_rights & chess.BB_SQUARES[sq])

    # -------------------------------------------------- pawns
    def advanced_enemy_pawn_attacks(self, max_rel_rank: int = 4) -> chess.SquareSet:
        """Squares hit by enemy pawns standing on our relative rank <= max_rel_rank (pawns pressing into our half)."""
        k = ("adv_pawn_att", max_rel_rank)
        if k not in self._x:
            s = chess.SquareSet()
            for x in self.b.pieces(chess.PAWN, self.them):
                if self.rel_rank(x) <= max_rel_rank:
                    s |= self.b.attacks(x)
            self._x[k] = s
        return self._x[k]

    def passer_path(self, sq: int, max_dist: int = 2) -> int:
        """sq is on the clear file ahead of a passed pawn (either side) within max_dist steps: max_dist + 1 - steps
        (2 = the very next square with the default), 0 otherwise. The road a passed pawn wants to run."""
        f, r = chess.square_file(sq), chess.square_rank(sq)
        best = 0
        for col, step in ((chess.WHITE, -1), (chess.BLACK, 1)):  # walk back from sq towards the pawn
            rr, dist = r + step, 1
            while 0 <= rr < 8 and dist <= max_dist:
                x = chess.square(f, rr)
                pc = self.b.piece_at(x)
                if pc is not None:
                    if pc.piece_type == chess.PAWN and pc.color == col and self.passed(x):
                        best = max(best, max_dist + 1 - dist)
                    break
                rr += step
                dist += 1
        return best

    # -------------------------------------------------- rank geometry
    def rank_neighbours(self, sq: int) -> tuple[int | None, int | None]:
        """Nearest occupied square to the left (towards the a-file) and right of sq on its rank (None to the edge)."""
        out = []
        r = chess.square_rank(sq)
        for df in (-1, 1):
            f, hit = chess.square_file(sq) + df, None
            while 0 <= f < 8:
                if self.b.piece_at(chess.square(f, r)) is not None:
                    hit = chess.square(f, r)
                    break
                f += df
            out.append(hit)
        return out[0], out[1]

    def rooks_connected(self, ours: bool = True) -> bool:
        """Two rooks of one side on the same rank with nothing between them (the back rank is cleared)."""
        rs = list(self.b.pieces(chess.ROOK, self.us if ours else self.them))
        return any(chess.square_rank(a) == chess.square_rank(b) and not (chess.between(a, b) & self.b.occupied)
                   for i, a in enumerate(rs) for b in rs[i + 1:])

    # -------------------------------------------------- Maia context (precomputed in info)
    def maia_entropy(self) -> float:
        """Entropy (nats) of Maia-3's top-5 move probabilities: ~0 when one move is obvious, ~1.6 when spread."""
        return -sum(q * math.log(q) for q in (self.info.get("maia_p5") or []) if q > 0)

    def maia_move_distance(self, sq: int) -> float:
        """Probability-weighted king-step distance from sq to Maia-3's top-5 moves (nearer of from/to square):
        0 when sq is on the move everyone expects, large when it is away from the action (3 with no Maia data)."""
        if "maia_moves" not in self._x:
            self._x["maia_moves"] = [(chess.Move.from_uci(u), q) for u, q in
                                     zip(self.info.get("maia_top5") or [], self.info.get("maia_p5") or [])]
        ms = self._x["maia_moves"]
        if not ms:
            return 3.0
        tot = sum(q for _, q in ms) or 1.0
        return sum(q * min(chess.square_distance(sq, m.from_square), chess.square_distance(sq, m.to_square))
                   for m, q in ms) / tot

    # ================================================== merged from detector batch 3 (2026-11-22)
    # -------------------------------------------------- development and game phase
    def on_home_square(self, sq: int, include_pawns: bool = True) -> bool:
        """The man on sq (either side) stands on its own starting square: an undeveloped piece, an unpushed pawn."""
        pc = self.b.piece_at(sq)
        if pc is None or (not include_pawns and pc.piece_type == chess.PAWN):
            return False
        return _START.piece_at(sq) == pc

    def opening_phase(self, span: int = 40) -> float:
        """1 at the first move, falling linearly with ply to 0 by ply `span` (default: 20 moves each)."""
        return max(0.0, 1.0 - self.ply() / span)

    def n_officers(self, ours: bool = True) -> int:
        """Knights, bishops, rooks and queens one side still has (7 at the start)."""
        c = self.us if ours else self.them
        return sum(len(self.b.pieces(t, c)) for t in _OFFICERS)

    # -------------------------------------------------- control
    def pawn_guards(self, sq: int, by_us: bool = True) -> int:
        """How many of one side's pawns attack sq (0, 1 or 2)."""
        return sum(self.b.piece_type_at(a) == chess.PAWN for a in self.attackers(sq, by_us))

    def weighted_control(self, sq: int, by_us: bool, w: dict[int, float]) -> float:
        """Sum of a per-piece-type weight over one side's attackers of sq (how firmly that side holds the square)."""
        return sum(w.get(self.b.piece_type_at(a), 0.0) for a in self.attackers(sq, by_us))

    def neighbour_control(self, sq: int, by_us: bool) -> float:
        """Fraction of the king-step neighbours of sq that one side attacks (who holds the area around sq)."""
        nb = chess.SquareSet(chess.BB_KING_ATTACKS[sq])
        return sum(self.n_guards(x, by_us) > 0 for x in nb) / len(nb)

    # -------------------------------------------------- pawns
    def pawn_blocked_by_own(self, sq: int) -> bool:
        """The pawn on sq (either side) has one of its own men directly in front of it: it cannot advance."""
        pc = self.b.piece_at(sq)
        if pc is None or pc.piece_type != chess.PAWN:
            return False
        front = sq + (8 if pc.color == chess.WHITE else -8)
        return 0 <= front < 64 and self.b.color_at(front) == pc.color

    # -------------------------------------------------- moves and plans
    def legal_from_count(self, sq: int) -> int:
        """Number of our legal moves that start on sq (true mobility of our man there; 0 for empty/enemy squares)."""
        if "legal_from" not in self._x:
            d: dict[int, int] = {}
            for m in self.b.legal_moves:
                d[m.from_square] = d.get(m.from_square, 0) + 1
            self._x["legal_from"] = d
        return self._x["legal_from"].get(sq, 0)

    def offers_trade(self, sq: int) -> bool:
        """We can move a knight/bishop/rook/queen to sq that an enemy piece of the same type attacks (a like-for-like
        trade offer)."""
        tt = self.guard_types(sq, by_us=False) - {chess.PAWN, chess.KING}
        return bool(tt) and any(self.b.piece_type_at(m.from_square) in tt for m in self.legal_to().get(sq, []))

    def in_our_pv(self, sq: int, moves: int = 3) -> bool:
        """sq is a from- or to-square of one of OUR first `moves` moves in Stockfish's PV (our plan, not their replies)."""
        for u in (self.info.get("sf_pv") or [])[0:2 * moves:2]:
            m = chess.Move.from_uci(u)
            if sq in (m.from_square, m.to_square):
                return True
        return False
