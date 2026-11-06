"""The canonical feature set: every per-ply feature name, with its compute tier, information regime and timing
(docs/SCIENCE.md §3). The analyzer and every baseline read features from here, so a name means one thing.

  tier     T0 free (moves, clocks, ratings) < T1 Maia-3 < T2 Stockfish 50k nodes < T3 deep (eval sets only).
           A derived feature takes the highest tier among its inputs.
  regime   R0 point (this position and game metadata) < R1 trajectory (this game up to here) < R2 career
           (the player's earlier games; not built).
  timing   relative to the move at ply k:
             pre   known before the move is chosen (position k, the clocks, history through ply k-1)
             act   the move itself and how long it took (this move's think time, its Maia-3 log-prob)
             post  needs the position after the move (CPL, wc loss, blunder): targets, never predictors of k
  table    "mov" (one row per ply, mover's point of view) or "pos" (one row per position, White's point of view)

Bump VERSION whenever a definition changes; results files record it.
"""

from __future__ import annotations

from dataclasses import dataclass

VERSION = "f1"
TIERS = ("T0", "T1", "T2", "T3")
REGIMES = ("R0", "R1", "R2")
TIMINGS = ("pre", "act", "post")


@dataclass(frozen=True)
class Feature:
    name: str
    tier: str
    regime: str
    timing: str
    table: str
    doc: str


# Identifiers and display fields: carried along, never features.
META = {"game_id", "mover", "uci", "san", "fen", "cls"}

_RAW = [
    # --- T0, mov ---
    ("elo_self", "T0", "R0", "pre", "mov", "mover's rating"),
    ("elo_opp", "T0", "R0", "pre", "mov", "opponent's rating"),
    ("base", "T0", "R0", "pre", "mov", "base time, s"),
    ("inc", "T0", "R0", "pre", "mov", "increment, s"),
    ("ply", "T0", "R0", "pre", "mov", "ply index (0 = White's first move)"),
    ("n_legal", "T0", "R0", "pre", "mov", "legal moves in the position"),
    ("forced", "T0", "R0", "pre", "mov", "only one legal move"),
    ("in_check", "T0", "R0", "pre", "mov", "mover is in check"),
    ("material_own", "T0", "R0", "pre", "mov", "material balance, mover's view (P1 N3 B3 R5 Q9)"),
    ("material_own_delta", "T0", "R1", "pre", "mov", "material change since the mover's previous decision"),
    ("clock_before", "T0", "R0", "pre", "mov", "mover's clock before the move, s"),
    ("clock_opp", "T0", "R0", "pre", "mov", "opponent's clock, s"),
    ("clock_frac", "T0", "R0", "pre", "mov", "mover's clock / base time"),
    ("clock_after", "T0", "R0", "act", "mov", "mover's clock after the move, s"),
    ("think_s", "T0", "R0", "act", "mov", "think time of this move, s (clock drop + increment)"),
    ("think_frac", "T0", "R0", "act", "mov", "logit((think + 0.5) / (clock before + inc))"),
    ("capture", "T0", "R0", "act", "mov", "the move captures"),
    ("gives_check", "T0", "R0", "act", "mov", "the move gives check"),
    ("promotion", "T0", "R0", "act", "mov", "the move promotes"),
    ("castle", "T0", "R0", "act", "mov", "the move castles"),
    ("tempo_median", "T0", "R1", "pre", "mov", "running median of the mover's think fraction before this move"),
    # --- T1, mov ---
    ("maia_entropy", "T1", "R0", "pre", "mov", "Maia-3 policy entropy over legal moves, nats"),
    ("maia_p_top", "T1", "R0", "pre", "mov", "Maia-3 probability of its top move"),
    ("maia_played_lp", "T1", "R0", "act", "mov", "Maia-3 log-prob of the played move"),
    ("maia_played_rank", "T1", "R0", "act", "mov", "rank of the played move under Maia-3 (1 = top; 17 = outside top 16 in the stored path)"),
    # --- T2, mov ---
    ("cp_own", "T2", "R0", "pre", "mov", "Stockfish cp before the move, mover's view, clipped to +/-1000"),
    ("wc_own", "T2", "R0", "pre", "mov", "Lichess winning chances before the move, mover's view"),
    ("cpl", "T2", "R0", "post", "mov", "centipawn loss, evals clipped to +/-1000"),
    ("wc_loss", "T2", "R0", "post", "mov", "drop in winning chances on this move, >= 0"),
    ("blunder", "T2", "R0", "post", "mov", "wc_loss >= 0.3 (Lichess)"),
    ("blunder2", "T2", "R0", "post", "mov", "wc_loss >= 0.2 (Maia's unit)"),
    # --- pos (White's view; pre for the move from that position) ---
    ("ply", "T0", "R0", "pre", "pos", "ply index of the position"),
    ("material_w", "T0", "R0", "pre", "pos", "material balance, White's view"),
    ("n_legal", "T0", "R0", "pre", "pos", "legal moves"),
    ("in_check", "T0", "R0", "pre", "pos", "side to move is in check"),
    ("material_vel", "T0", "R1", "pre", "pos", "material change since the previous position"),
    ("maia_win_w", "T1", "R0", "pre", "pos", "Maia-3 P(White wins)"),
    ("maia_draw", "T1", "R0", "pre", "pos", "Maia-3 P(draw)"),
    ("maia_loss_w", "T1", "R0", "pre", "pos", "Maia-3 P(White loses)"),
    ("maia_es_w", "T1", "R0", "pre", "pos", "Maia-3 expected score for White"),
    ("sharpness", "T1", "R0", "pre", "pos", "jk_182 sharpness from Maia-3 W/D/L, capped at 50 (saturates at 1500)"),
    ("cp_w", "T2", "R0", "pre", "pos", "Stockfish cp, White's view, mate = +/-10000"),
    ("wc_w", "T2", "R0", "pre", "pos", "Lichess winning chances, White's view"),
]

# Per-side transforms over own moves (transforms.SIDE_CHANNELS x SIDE_SUFFIXES). The plain transform includes the
# current move, so it inherits the channel's timing; "_prev" is its value as of the mover's previous move (pre).
SIDE_CHANNELS = ("think_s", "think_frac", "clock_frac", "maia_entropy", "maia_played_lp", "cpl", "wc_loss")
SIDE_SUFFIXES = ("ewma", "roll_mean", "roll_std", "vel")
# Position transforms (transforms.position_transforms); on the mov table _white/_black become _own.
POS_SERIES = (("wc_w", "T2"), ("maia_es_w", "T1"))
POS_SUFFIXES = ("volatility", "vel", "drawdown_white", "drawdown_black")


def _build() -> dict[tuple[str, str], Feature]:
    reg = {(n, t): Feature(n, tier, reg, tim, t, doc) for n, tier, reg, tim, t, doc in _RAW}
    mov = {n: f for (n, t), f in reg.items() if t == "mov"}
    for ch in SIDE_CHANNELS:
        base = mov[ch]
        for suf in SIDE_SUFFIXES:
            reg[(f"{ch}_{suf}", "mov")] = Feature(f"{ch}_{suf}", base.tier, "R1", base.timing, "mov",
                                                  f"{suf} of {ch} over the mover's own moves, through this move")
            reg[(f"{ch}_{suf}_prev", "mov")] = Feature(f"{ch}_{suf}_prev", base.tier, "R1", "pre", "mov",
                                                       f"{suf} of {ch} through the mover's previous move")
    for ch, tier in POS_SERIES:
        for suf in POS_SUFFIXES:
            reg[(f"{ch}_{suf}", "pos")] = Feature(f"{ch}_{suf}", tier, "R1", "pre", "pos", f"{suf} of {ch} win %")
        for suf in ("volatility", "vel_own", "drawdown_own"):
            reg[(f"{ch}_{suf}", "mov")] = Feature(f"{ch}_{suf}", tier, "R1", "pre", "mov",
                                                  f"{suf} of {ch} win % at the position the move is chosen from")
    return reg


REGISTRY = _build()


def get(name: str, table: str = "mov") -> Feature:
    return REGISTRY[(name, table)]


def select(table: str = "mov", tier: str = "T3", regime: str = "R1", timing: tuple[str, ...] = ("pre",)) -> list[str]:
    """Feature names a baseline may use: at or below a tier and regime, with the given timings."""
    return [f.name for f in REGISTRY.values()
            if f.table == table and TIERS.index(f.tier) <= TIERS.index(tier)
            and REGIMES.index(f.regime) <= REGIMES.index(regime) and f.timing in timing]


def unregistered(names, table: str) -> list[str]:
    return sorted(n for n in names if n not in META and (n, table) not in REGISTRY)
