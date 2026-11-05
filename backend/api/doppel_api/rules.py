"""Game rules on a Game row: legality, clocks, endings. Pure functions of (game, now); no I/O."""

from __future__ import annotations

import chess
import numpy as np

from .db import Game


class IllegalMove(ValueError):
    pass


def board(g: Game) -> chess.Board:
    b = chess.Board()
    for u in g.moves:
        b.push_uci(u)
    return b


def side_to_move(g: Game) -> str:
    return "white" if len(g.moves) % 2 == 0 else "black"


def _end(g: Game, result: str, termination: str) -> None:
    g.status, g.result, g.termination, g.turn_started = "over", result, termination, None


def _flag(g: Game, loser: str) -> None:
    # Lichess rule: flagging against a lone king (or no mating material) is a draw.
    winner = chess.WHITE if loser == "black" else chess.BLACK
    if board(g).has_insufficient_material(winner):
        _end(g, "1/2-1/2", "timeout_vs_insufficient")
    else:
        _end(g, "0-1" if loser == "white" else "1-0", "timeout")


def remaining_ms(g: Game, color: str, now: float) -> int:
    ms = g.white_ms if color == "white" else g.black_ms
    if g.status == "active" and g.turn_started is not None and side_to_move(g) == color:
        ms -= int((now - g.turn_started) * 1000)
    return ms


def settle(g: Game, now: float) -> bool:
    """Apply a flag that has happened since the last move. Returns True if the row changed."""
    if g.status != "active" or g.turn_started is None:
        return False
    mover = side_to_move(g)
    if remaining_ms(g, mover, now) <= 0:
        setattr(g, f"{mover}_ms", 0)
        _flag(g, mover)
        return True
    return False


def play(g: Game, uci: str, now: float, charge_ms: int | None = None) -> dict:
    """Validate and apply one move by the side to move, charging its clock: wall time since the turn began,
    or `charge_ms` instead (a bot's sampled think time). Returns the move's timing record."""
    if settle(g, now) or g.status != "active":
        raise IllegalMove("game is over")
    b = board(g)
    try:
        mv = b.parse_uci(uci)  # also accepts king-takes-rook castling (e1h1), as some clients send it
    except ValueError:
        raise IllegalMove(f"illegal move {uci!r}")
    if not mv:  # parse_uci lets the null move "0000" through
        raise IllegalMove("null move")
    uci = mv.uci()  # store the canonical form (e1g1)
    mover = side_to_move(g)
    began = g.turn_started if g.turn_started is not None else g.last_move_at
    wall = int((now - began) * 1000) if began is not None else 0
    spent = 0
    if g.turn_started is not None:
        spent = wall if charge_ms is None else charge_ms
        setattr(g, f"{mover}_ms", getattr(g, f"{mover}_ms") - spent + g.inc_ms)
    b.push(mv)
    g.moves = [*g.moves, uci]  # reassign so the JSON column is marked dirty
    # Clocks start once both sides have made a move (Lichess convention).
    g.turn_started = now if len(g.moves) >= 2 else None
    g.last_move_at = now
    out = b.outcome(claim_draw=True)
    if out is not None:
        _end(g, out.result(), out.termination.name.lower())
    return {"uci": uci, "at": now, "wall_ms": wall, "spent_ms": spent,
            "clock_after_ms": getattr(g, f"{mover}_ms")}


def resign(g: Game, color: str, now: float) -> None:
    if settle(g, now) or g.status != "active":
        raise IllegalMove("game is over")
    _end(g, "0-1" if color == "white" else "1-0", "resign")


def sample_z(prior: dict | None, rng: np.random.Generator) -> list | None:
    """Phase 1 decision: z is drawn once per game (nested under a per-player z, later). None for stateless bots."""
    if not prior:
        return None
    if prior.get("type") == "normal":
        return rng.normal(0.0, prior.get("scale", 1.0), int(prior["dim"])).round(5).tolist()
    raise ValueError(f"unknown z prior {prior!r}")


def think_ms(g: Game, p_top: float, now: float, rng: np.random.Generator) -> int:
    """Human-like think time for the bot's next move (placeholder until a model has a clock head).

    Quick for the first few moves and for forced moves; otherwise its remaining time spread over the moves it
    expects to play, scaled up when the policy is unsure (low top-move probability), with lognormal noise.
    Never spends more than a quarter of what is left, and leaves a margin so it cannot flag by thinking."""
    mover = side_to_move(g)
    left = remaining_ms(g, mover, now)
    own = len(g.moves) // 2
    if board(g).legal_moves.count() == 1:
        t = rng.uniform(300, 900)
    elif own < 3:
        t = rng.uniform(400, 1500)
    else:
        budget = left / max(50 - own, 25) + 0.6 * g.inc_ms
        t = budget * (0.3 + 1.3 * (1.0 - p_top)) * rng.lognormal(0.0, 0.55)
    cap = max(150.0, min(0.25 * left, left - 1500.0))
    return int(np.clip(t, 150.0, cap))
