"""Accounts and consent.

Stage 0 accounts are anonymous: the device calls POST /api/users once, keeps the bearer token, and sends it
as `Authorization: Bearer <token>` (WebSocket: `?token=`). Only the token's sha256 is stored. Passing the
device's install_id claims the games it played before accounts existed.

Training consent is opt-in, versioned, withdrawable and 18+ only (self-declared, and refused for Lichess
kid-mode accounts). Only games of users who hold it with the 18+ flag at export time are exported, so
training-v1 grants, made before the age check, no longer count; withdrawing turns that user's previously
exported games into tombstones (see export.py).
Deleting the account removes its games, plies, consents, ratings, tokens and imported Lichess history.

History consent (history-v1, history.py) is a separate opt-in for importing the user's public Lichess games; it
records a self-declared 18+ flag, and withdrawing it deletes the imported games.

Signing in with Lichess (lichess_login.py) attaches a verified Lichess id to the account and lets more devices
sign in to it; each device gets its own token.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import time

from fastapi import APIRouter, HTTPException, Request, WebSocket
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, update

from .db import AuthToken, Consent, Game, Ply, Rating, User, delete_history

CONSENT_VERSION = "training-v2"
CONSENT_TEXT = (
    "Help train doppels. If you turn this on, the games you play here (moves, clock times and the ratings "
    "involved) may be used to train and evaluate Doppel chess models. Your account is replaced by a random "
    "id in every training export, so the research side never sees who you are. You can turn it off at any "
    "time; games already exported are then deleted from training data at the next export. You must be 18 or older.")

HISTORY_VERSION = "history-v1"
HISTORY_TEXT = (
    "Import my Lichess games. If you turn this on, Doppel downloads your public Lichess games (moves, clock "
    "times, ratings, openings and results; not your opponents' names), stores them, and uses them to train a "
    "personal doppel that plays like you, for you to play against. Only if you have also turned on 'Help train "
    "doppels' may they also be used to train general Doppel models. You must be 18 or older. You can turn this "
    "off at any time: turning it off, or deleting your account, deletes the imported games.")

router = APIRouter()


_TOKEN_IN_URL = re.compile(r"([?&](?:token|code|state|login_code)=)[^&\s\"]+")


class RedactTokens(logging.Filter):
    """Keeps secrets out of server logs: the WebSocket carries a bearer token in its URL (?token=), the Lichess
    sign-in callback carries an OAuth code and state, and uvicorn logs request lines verbatim."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "=" in msg and _TOKEN_IN_URL.search(msg):
            if isinstance(record.args, tuple) and record.args:
                # uvicorn's access formatter unpacks args (client, method, path, ...): redact inside them
                record.args = tuple(_TOKEN_IN_URL.sub(r"\1REDACTED", a) if isinstance(a, str) else a
                                    for a in record.args)
            if isinstance(record.msg, str):
                record.msg = _TOKEN_IN_URL.sub(r"\1REDACTED", record.msg)
        return True


def install_log_redaction() -> None:
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, RedactTokens) for f in lg.filters):
            lg.addFilter(RedactTokens())


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def require_auth() -> bool:
    """DOPPEL_REQUIRE_AUTH=1 refuses anonymous game creation (keep off until every client has an account)."""
    return os.environ.get("DOPPEL_REQUIRE_AUTH", "0") == "1"


def user_from_token(sess, token: str | None) -> User | None:
    if not token:
        return None
    h = hash_token(token)
    u = sess.scalar(select(User).where(User.token_hash == h))
    if u is None:
        t = sess.get(AuthToken, h)
        u = sess.get(User, t.user_id) if t is not None else None
    if u is None:
        raise HTTPException(401, "unknown token")
    return u


def bearer(request: Request | WebSocket) -> str | None:
    h = request.headers.get("authorization", "")
    if h.lower().startswith("bearer "):
        return h[7:].strip()
    return request.query_params.get("token")


def current_user(sess, request: Request | WebSocket) -> User | None:
    return user_from_token(sess, bearer(request))


def authorize(g: Game, user: User | None) -> None:
    """A game created by an account can be read and played only with that account's token."""
    if g.user_id is not None and (user is None or user.id != g.user_id):
        raise HTTPException(403, "not your game")


def holds_consent(sess, user_id: str, kind: str = "training") -> Consent | None:
    c = sess.scalar(select(Consent).where(Consent.user_id == user_id, Consent.kind == kind)
                    .order_by(Consent.granted_at.desc(), Consent.id.desc()).limit(1))
    return c if c is not None and c.withdrawn_at is None else None


def training_consent(sess, user_id: str) -> Consent | None:
    """The training consent that opens the export: held, granted with the 18+ flag, and the account not
    (or no longer) in Lichess kid mode."""
    c = holds_consent(sess, user_id)
    if c is None or not c.adult:
        return None
    u = sess.get(User, user_id)
    return c if u is not None and not u.lichess_kid else None


def me(sess, u: User) -> dict:
    c = training_consent(sess, u.id)
    h = holds_consent(sess, u.id, "history")
    r = sess.get(Rating, f"user:{u.id}")
    n_games = len(sess.scalars(select(Game.id).where(Game.user_id == u.id)).all())
    return {
        "id": u.id, "display_name": u.display_name, "created_at": u.created_at, "games": n_games,
        "lichess": {"id": u.lichess_id, "username": u.lichess_username} if u.lichess_id else None,
        "consent": {"training": c is not None, "version": c.version if c else None,
                    "current_version": CONSENT_VERSION},
        "history_consent": {"granted": h is not None, "version": h.version if h else None,
                            "adult": bool(h.adult) if h else False, "current_version": HISTORY_VERSION},
        "rating": {"rating": r.rating, "rd": r.rd, "games": r.games} if r else None,
    }


class NewUser(BaseModel):
    install_id: str | None = Field(None, max_length=64)
    display_name: str | None = Field(None, max_length=40)


class ConsentIntent(BaseModel):
    training: bool
    version: str = CONSENT_VERSION
    adult: bool = False   # "I am 18 or older", ticked by the user


class Profile(BaseModel):
    display_name: str | None = Field(None, max_length=40)


@router.post("/api/users")
def create_user(req: NewUser, request: Request):
    """New anonymous account. The token is returned once and never again."""
    token = secrets.token_urlsafe(32)
    with request.app.state.sessions() as sess:
        if req.install_id and sess.scalar(select(User).where(User.install_id == req.install_id)):
            raise HTTPException(409, "install id already has an account")
        u = User(token_hash=hash_token(token), install_id=req.install_id, display_name=req.display_name)
        sess.add(u)
        sess.flush()
        claimed = 0
        if req.install_id:
            claimed = sess.execute(update(Game).where(Game.player == req.install_id, Game.user_id.is_(None))
                                   .values(user_id=u.id)).rowcount
        sess.commit()
        return {"token": token, "claimed_games": claimed, "user": me(sess, u)}


def _me_or_401(sess, request: Request) -> User:
    u = current_user(sess, request)
    if u is None:
        raise HTTPException(401, "sign in first")
    return u


@router.get("/api/me")
def get_me(request: Request):
    with request.app.state.sessions() as sess:
        return me(sess, _me_or_401(sess, request))


@router.patch("/api/me")
def patch_me(req: Profile, request: Request):
    with request.app.state.sessions() as sess:
        u = _me_or_401(sess, request)
        u.display_name = req.display_name
        sess.commit()
        return me(sess, u)


@router.get("/api/consent")
def consent_text():
    return {"version": CONSENT_VERSION, "training": CONSENT_TEXT,
            "history_version": HISTORY_VERSION, "history": HISTORY_TEXT}


@router.put("/api/me/consent")
def put_consent(req: ConsentIntent, request: Request):
    with request.app.state.sessions() as sess:
        u = _me_or_401(sess, request)
        held = holds_consent(sess, u.id)
        if req.training:
            if u.lichess_kid:
                raise HTTPException(403, "Lichess kid-mode accounts can't help train doppels.")
            if req.version != CONSENT_VERSION:
                raise HTTPException(409, f"consent text changed: show version {CONSENT_VERSION}")
            if not req.adult:
                raise HTTPException(422, "You must be 18 or older to help train doppels.")
            if held is None or held.version != req.version or not held.adult:
                if held is not None:
                    held.withdrawn_at = time.time()
                sess.add(Consent(user_id=u.id, kind="training", version=req.version, adult=True))
        elif held is not None:
            held.withdrawn_at = time.time()
        sess.commit()
        return me(sess, u)


@router.delete("/api/me")
async def delete_me(request: Request):
    """Delete the account and everything it played, and its imported Lichess history. Games already exported
    become tombstones at the next export (the export ledger remembers their ids, nothing else).
    Async on purpose: it runs on the event loop, like the history import's writes, so a running import is
    cancelled and purged with no batch slipping in between."""
    with request.app.state.sessions() as sess:
        u = _me_or_401(sess, request)
        importer = getattr(request.app.state, "history", None)
        if importer is not None:
            importer.cancel(u.id)
        imported = delete_history(sess, u.id)
        ids = sess.scalars(select(Game.id).where(Game.user_id == u.id)).all()
        sess.execute(delete(Ply).where(Ply.game_id.in_(ids)))
        sess.execute(delete(Game).where(Game.id.in_(ids)))
        sess.execute(delete(Rating).where(Rating.subject == f"user:{u.id}"))
        sess.execute(delete(Consent).where(Consent.user_id == u.id))
        sess.execute(delete(AuthToken).where(AuthToken.user_id == u.id))
        sess.delete(u)
        sess.commit()
        return {"deleted_games": len(ids), "deleted_imported_games": imported}
