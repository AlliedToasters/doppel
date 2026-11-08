"""Storage. SQLite file for Stage 0; set DOPPEL_DB_URL=postgresql+psycopg://... for Stage 1. Same schema."""

from __future__ import annotations

import os
import time
import uuid

from sqlalchemy import (JSON, BigInteger, Boolean, Float, ForeignKey, Index, Integer, String, UniqueConstraint,
                        create_engine, delete, inspect, text)
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
    player: Mapped[str | None] = mapped_column(String(64), nullable=True)  # device install id (legacy, pre-accounts)
    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True, index=True)
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


class User(Base):
    """An account. Stage 0: anonymous, created by the device on first launch and held by a bearer token
    (only its sha256 is stored). A Lichess login can be linked to it later without changing the id."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    install_id: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(40), nullable=True)
    # Signed in with Lichess (identity only: no Lichess token is kept). lichess_id is Lichess's stable lowercase id.
    lichess_id: Mapped[str | None] = mapped_column(String(32), nullable=True, unique=True, index=True)
    lichess_username: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Lichess kid mode, read once at sign-in (preference:read). None = signed in before we asked: sign in again.
    lichess_kid: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


class AuthToken(Base):
    """Extra bearer tokens for an account (one per signed-in device); the first device's token is User.token_hash.
    Only sha256 hashes are stored."""

    __tablename__ = "auth_tokens"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class OAuthPending(Base):
    """A Lichess sign-in in flight: the PKCE verifier for its state, kept 10 minutes."""

    __tablename__ = "oauth_pending"

    state: Mapped[str] = mapped_column(String(64), primary_key=True)
    verifier: Mapped[str] = mapped_column(String(128))
    client: Mapped[str] = mapped_column(String(8))  # "web" | "app": where the callback sends the browser
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class LoginCode(Base):
    """A verified Lichess identity waiting to be exchanged for a Doppel token (one use, 2 minutes). Codes go
    to the client in a redirect URL, so only their hash is stored and the token itself never appears in a URL."""

    __tablename__ = "login_codes"

    code_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    lichess_id: Mapped[str] = mapped_column(String(32))
    lichess_username: Mapped[str] = mapped_column(String(32))
    lichess_kid: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class Consent(Base):
    """Consent history, append-only: granting adds a row, withdrawing stamps withdrawn_at. A user holds a
    consent kind while their latest row of that kind is not withdrawn."""

    __tablename__ = "consents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # "training" | "history"
    version: Mapped[str] = mapped_column(String(32))  # the consent text the user saw
    granted_at: Mapped[float] = mapped_column(Float, default=time.time)
    withdrawn_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    adult: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # self-declared 18+ when granted (history)


class ImportedGame(Base):
    """A game imported from the user's public Lichess history (history-v1 consent; history.py). Kept only while
    that consent is held: withdrawing it or deleting the account deletes these rows. The opponent is not stored."""

    __tablename__ = "imported_games"
    __table_args__ = (UniqueConstraint("user_id", "lichess_game_id", name="uq_imported_user_game"),
                      Index("ix_imported_user_played", "user_id", "played_at"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"))
    lichess_game_id: Mapped[str] = mapped_column(String(16))
    played_at: Mapped[float] = mapped_column(Float)  # epoch s (Lichess createdAt)
    speed: Mapped[str] = mapped_column(String(16))  # ultraBullet | bullet | blitz | rapid | classical | correspondence
    perf: Mapped[str] = mapped_column(String(16))
    rated: Mapped[bool] = mapped_column(Boolean)
    color: Mapped[str] = mapped_column(String(5))  # the user's colour
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)  # the user's rating when the game began
    opp_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)  # None vs the Lichess AI
    result: Mapped[str | None] = mapped_column(String(7), nullable=True)  # "1-0" | "0-1" | "1/2-1/2"
    status: Mapped[str] = mapped_column(String(24))  # Lichess termination: mate, resign, outoftime, draw, ...
    base_s: Mapped[int | None] = mapped_column(Integer, nullable=True)  # None for correspondence
    inc_s: Mapped[int | None] = mapped_column(Integer, nullable=True)
    moves: Mapped[list] = mapped_column(JSON, default=list)  # SAN, as the export gives them
    clocks: Mapped[list | None] = mapped_column(JSON, nullable=True)  # mover's clock after each ply, centiseconds
    opening_eco: Mapped[str | None] = mapped_column(String(8), nullable=True)
    opening_name: Mapped[str | None] = mapped_column(String(128), nullable=True)


class HistoryImport(Base):
    """One row per user: the state of their Lichess history import."""

    __tablename__ = "history_imports"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    state: Mapped[str] = mapped_column(String(8))  # "queued" | "running" | "done" | "error"
    added: Mapped[int] = mapped_column(Integer, default=0)  # games added by the latest run
    error: Mapped[str | None] = mapped_column(String(200), nullable=True)
    started_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)
    # createdAt (ms) of the newest stored game after the last completed run: the next run's `since`.
    cursor_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class Rating(Base):
    """Current ratings. subject is "user:<id>" or "bot:<package_id>@<bot_elo>" (a package played at a given
    Elo setting is a distinct opponent). Elo scale; rd is the posterior standard deviation."""

    __tablename__ = "ratings"

    subject: Mapped[str] = mapped_column(String(120), primary_key=True)
    rating: Mapped[float] = mapped_column(Float)
    rd: Mapped[float] = mapped_column(Float)
    games: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)


class Exported(Base):
    """Export ledger: which game ids the lab has received, so a withdrawn or deleted game can be sent as a
    tombstone. Holds no user data; game_id is deliberately not a foreign key (the game may be gone)."""

    __tablename__ = "exported"

    game_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    exported_at: Mapped[float] = mapped_column(Float)
    tombstoned_at: Mapped[float | None] = mapped_column(Float, nullable=True)


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


def delete_history(sess, user_id: str) -> int:
    """Delete a user's imported Lichess games and import status; returns the number of games deleted."""
    n = sess.execute(delete(ImportedGame).where(ImportedGame.user_id == user_id)).rowcount
    sess.execute(delete(HistoryImport).where(HistoryImport.user_id == user_id))
    return n


_ADDED = [  # additive columns for databases created by an earlier skeleton
    ("bot_pace", "VARCHAR(8) NOT NULL DEFAULT 'realtime'"),
    ("player", "VARCHAR(64)"),
    ("last_move_at", "FLOAT"),
    ("user_id", "VARCHAR(36) REFERENCES users(id)"),
]


_ADDED_USERS = [("lichess_id", "VARCHAR(32)"), ("lichess_username", "VARCHAR(32)"), ("lichess_kid", "BOOLEAN")]
_ADDED_BY_TABLE = {"games": _ADDED, "users": _ADDED_USERS, "login_codes": [("lichess_kid", "BOOLEAN")],
                   "consents": [("adult", "BOOLEAN")]}  # nullable, no default: safe on Postgres and SQLite


def make_sessionmaker(url: str | None = None):
    url = url or os.environ.get("DOPPEL_DB_URL", "sqlite:///data/api.db")
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    engine = create_engine(url, **kw)
    Base.metadata.create_all(engine)  # skeleton: additive columns only until the schema settles (then Alembic)
    insp = inspect(engine)
    have = {t: {c["name"] for c in insp.get_columns(t)} for t in _ADDED_BY_TABLE}
    with engine.begin() as conn:
        for table, cols in _ADDED_BY_TABLE.items():
            for col, ddl in cols:
                if col not in have[table]:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_lichess_id ON users (lichess_id)"))
    return sessionmaker(engine, expire_on_commit=False)
