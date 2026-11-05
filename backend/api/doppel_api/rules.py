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


def play(g: Game, uci: str, now: float) -> None:
    """Validate and apply one move by the side to move, charging its clock."""
    if settle(g, now) or g.status != "active":
        raise IllegalMove("game is over")
    b = board(g)
    try:
        mv = chess.Move.from_uci(uci)
    except ValueError:
        raise IllegalMove(f"bad uci {uci!r}")
    if mv not in b.legal_moves:
        raise IllegalMove(f"illegal move {uci}")
    mover = side_to_move(g)
    if g.turn_started is not None:
        spent = int((now - g.turn_started) * 1000)
        setattr(g, f"{mover}_ms", getattr(g, f"{mover}_ms") - spent + g.inc_ms)
    b.push(mv)
    g.moves = [*g.moves, uci]  # reassign so the JSON column is marked dirty
    # Clocks start once both sides have made a move (Lichess convention).
    g.turn_started = now if len(g.moves) >= 2 else None
    out = b.outcome(claim_draw=True)
    if out is not None:
        _end(g, out.result(), out.termination.name.lower())


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
