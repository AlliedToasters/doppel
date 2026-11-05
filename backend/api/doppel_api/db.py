"""Storage. SQLite file for Stage 0; set DOPPEL_DB_URL=postgresql+psycopg://... for Stage 1. Same schema."""

from __future__ import annotations

import os
import time
import uuid

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Game(Base):
    """One Doppel-hosted game. The server holds the moves, the clocks and the doppel's z, so a game survives
    a client restart and resumes with the same z (ARCHITECTURE.md § backend)."""

    __tablename__ = "games"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    bot: Mapped[str] = mapped_column(String(64))
    package_id: Mapped[str] = mapped_column(String(80))  # rated player = package id
    bot_elo: Mapped[int] = mapped_column(Integer)
    human_elo: Mapped[int] = mapped_column(Integer)
    human_color: Mapped[str] = mapped_column(String(5))  # "white" | "black"
    player: Mapped[str | None] = mapped_column(String(64), nullable=True)  # anonymous install id (no accounts yet)
    base_ms: Mapped[int] = mapped_column(Integer)
    inc_ms: Mapped[int] = mapped_column(Integer)
    white_ms: Mapped[int] = mapped_column(Integer)
    black_ms: Mapped[int] = mapped_column(Integer)
    turn_started: Mapped[float | None] = mapped_column(Float, nullable=True)  # epoch s; None = clocks not running
    last_move_at: Mapped[float | None] = mapped_column(Float, nullable=True)  # epoch s; wall time base for timing
    moves: Mapped[list] = mapped_column(JSON, default=list)
    z: Mapped[list | None] = mapped_column(JSON, nullable=True)  # per-game z, sampled once at creation
    # "realtime": the bot waits its think time on the wall clock. "fast": it replies at once but the same
    # think time is still charged to its clock. Either way the clocks read as a human game.
    bot_pace: Mapped[str] = mapped_column(String(8), default="realtime")
    status: Mapped[str] = mapped_column(String(8), default="active")  # "active" | "over"
    result: Mapped[str | None] = mapped_column(String(7), nullable=True)  # "1-0" | "0-1" | "1/2-1/2"
    termination: Mapped[str | None] = mapped_column(String(24), nullable=True)


class Ply(Base):
    """One move with its timing: the native corpus. Human times are real; bot times are synthetic until a model
    has a clock head (see think_source), so the lab must never train a think-time head on actor="bot" rows."""

    __tablename__ = "plies"

    game_id: Mapped[str] = mapped_column(String(36), ForeignKey("games.id"), primary_key=True)
    ply: Mapped[int] = mapped_column(Integer, primary_key=True)
    uci: Mapped[str] = mapped_column(String(5))
    actor: Mapped[str] = mapped_column(String(5))  # "human" | "bot"
    at: Mapped[float] = mapped_column(Float)  # server epoch s when applied
    wall_ms: Mapped[int] = mapped_column(Integer)  # server wall time since the turn began (incl. network)
    spent_ms: Mapped[int] = mapped_column(Integer)  # charged to the clock (0 before clocks start)
    clock_after_ms: Mapped[int] = mapped_column(Integer)  # mover's clock after the move, increment included
    client_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)  # think time measured on the device
    premove: Mapped[bool] = mapped_column(Boolean, default=False)
    think_source: Mapped[str] = mapped_column(String(24))  # "human" | "heuristic_v1" | "model:<package>"
    pace: Mapped[str] = mapped_column(String(8))  # game pace when played


_ADDED = [  # additive columns for databases created by an earlier skeleton
    ("bot_pace", "VARCHAR(8) NOT NULL DEFAULT 'realtime'"),
    ("player", "VARCHAR(64)"),
    ("last_move_at", "FLOAT"),
]


def make_sessionmaker(url: str | None = None):
    url = url or os.environ.get("DOPPEL_DB_URL", "sqlite:///data/api.db")
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    engine = create_engine(url, **kw)
    Base.metadata.create_all(engine)  # skeleton: additive columns only until the schema settles (then Alembic)
    have = {c["name"] for c in inspect(engine).get_columns("games")}
    with engine.begin() as conn:
        for col, ddl in _ADDED:
            if col not in have:
                conn.execute(text(f"ALTER TABLE games ADD COLUMN {col} {ddl}"))
    return sessionmaker(engine, expire_on_commit=False)
