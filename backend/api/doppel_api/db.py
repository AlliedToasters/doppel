"""Storage. SQLite file for Stage 0; set DOPPEL_DB_URL=postgresql+psycopg://... for Stage 1. Same schema."""

from __future__ import annotations

import os
import time
import uuid

from sqlalchemy import JSON, Float, Integer, String, create_engine
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
    base_ms: Mapped[int] = mapped_column(Integer)
    inc_ms: Mapped[int] = mapped_column(Integer)
    white_ms: Mapped[int] = mapped_column(Integer)
    black_ms: Mapped[int] = mapped_column(Integer)
    turn_started: Mapped[float | None] = mapped_column(Float, nullable=True)  # epoch s; None = clocks not running
    moves: Mapped[list] = mapped_column(JSON, default=list)
    z: Mapped[list | None] = mapped_column(JSON, nullable=True)  # per-game z, sampled once at creation
    status: Mapped[str] = mapped_column(String(8), default="active")  # "active" | "over"
    result: Mapped[str | None] = mapped_column(String(7), nullable=True)  # "1-0" | "0-1" | "1/2-1/2"
    termination: Mapped[str | None] = mapped_column(String(24), nullable=True)


def make_sessionmaker(url: str | None = None):
    url = url or os.environ.get("DOPPEL_DB_URL", "sqlite:///data/api.db")
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    engine = create_engine(url, **kw)
    Base.metadata.create_all(engine)  # skeleton: no migrations until the schema settles (then Alembic)
    return sessionmaker(engine, expire_on_commit=False)
