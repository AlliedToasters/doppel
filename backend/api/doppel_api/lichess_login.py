"""Sign in with Lichess: identity only, run by the server.

    GET  /api/auth/lichess/start?client=web|app   302 to lichess.org/oauth (PKCE, no scopes)
    GET  /api/auth/lichess/callback               Lichess sends the browser back here
    POST /api/auth/lichess/exchange {code}        one-time login code -> Doppel bearer token

The callback exchanges the code for a Lichess token, reads GET /api/account once (id and username), and revokes
the token straight away, so the backend never holds a Lichess token. This is separate from the client's own
board:play login, which stays on the device (the Lichess hard boundary in ARCHITECTURE.md).

The browser then goes to the client with a one-time login code (web: {DOPPEL_PUBLIC_URL}/account/?login_code=...,
app: app.doppelchess.doppel://doppel-login?login_code=...); the client posts it to /exchange, sending its current
anonymous token if it has one:
  - Lichess id new, no token or an anonymous token: the account gets the Lichess id (an anonymous account keeps
    its id and games).
  - Lichess id already known: that account is returned with a new token for this device. An anonymous
    account's games move into it and the anonymous account is deleted. Its consents are not carried over, so
    the moved games follow the existing account's consent.
  - The client's account is linked to a different Lichess id: 409 (sign out first).
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import time
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, update

from .accounts import current_user, hash_token, me
from .db import AuthToken, Consent, Game, LoginCode, OAuthPending, Rating, User

LICHESS = os.environ.get("DOPPEL_LICHESS_URL", "https://lichess.org")
CLIENT_ID = os.environ.get("DOPPEL_LICHESS_CLIENT_ID", "doppelbot.org")
APP_TARGET = "app.doppelchess.doppel://doppel-login"
PENDING_S, CODE_S = 600, 120

router = APIRouter()


def public_url() -> str:
    return os.environ.get("DOPPEL_PUBLIC_URL", "http://localhost:8612").rstrip("/")


def redirect_uri() -> str:
    return f"{public_url()}/api/auth/lichess/callback"


def _target(client: str) -> str:
    return APP_TARGET if client == "app" else f"{public_url()}/account/"


def _back(client: str, **q) -> RedirectResponse:
    return RedirectResponse(f"{_target(client)}?{urlencode(q)}", status_code=302)


@router.get("/api/auth/lichess/start")
def start(request: Request, client: str = "web"):
    if client not in ("web", "app"):
        raise HTTPException(422, "client must be web or app")
    state, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    with request.app.state.sessions() as sess:
        sess.execute(delete(OAuthPending).where(OAuthPending.created_at < time.time() - PENDING_S))
        sess.add(OAuthPending(state=state, verifier=verifier, client=client))
        sess.commit()
    q = {"response_type": "code", "client_id": CLIENT_ID, "redirect_uri": redirect_uri(), "scope": "",
         "code_challenge_method": "S256", "code_challenge": challenge, "state": state}
    return RedirectResponse(f"{LICHESS}/oauth?{urlencode(q)}", status_code=302)


async def _identity(lc: httpx.AsyncClient, code: str, verifier: str) -> tuple[str, str]:
    """Code -> token -> (id, username); the token is revoked before returning."""
    r = await lc.post("/api/token", data={"grant_type": "authorization_code", "code": code,
                                          "code_verifier": verifier, "redirect_uri": redirect_uri(),
                                          "client_id": CLIENT_ID})
    r.raise_for_status()
    tok = r.json()["access_token"]
    h = {"authorization": f"Bearer {tok}"}
    try:
        a = await lc.get("/api/account", headers=h)
        a.raise_for_status()
        acct = a.json()
    finally:
        try:
            await lc.delete("/api/token", headers=h)
        except httpx.HTTPError:
            pass   # the token expires on its own; never fail a login over the revoke
    return acct["id"], acct["username"]


@router.get("/api/auth/lichess/callback")
async def callback(request: Request, state: str = "", code: str | None = None, error: str | None = None):
    with request.app.state.sessions() as sess:
        p = sess.get(OAuthPending, state) if state else None
        if p is None or p.created_at < time.time() - PENDING_S:
            raise HTTPException(400, "unknown or expired sign-in; start again")
        client, verifier = p.client, p.verifier
        sess.delete(p)
        sess.commit()
    if error or not code:
        return _back(client, login_error=error or "no_code")
    try:
        lid, uname = await _identity(request.app.state.lichess, code, verifier)
    except (httpx.HTTPError, KeyError, ValueError):
        return _back(client, login_error="lichess")
    login_code = secrets.token_urlsafe(32)
    with request.app.state.sessions() as sess:
        sess.execute(delete(LoginCode).where(LoginCode.created_at < time.time() - CODE_S))
        sess.add(LoginCode(code_hash=hash_token(login_code), lichess_id=lid, lichess_username=uname))
        sess.commit()
    return _back(client, login_code=login_code)


class Exchange(BaseModel):
    code: str = Field(..., max_length=128)


@router.post("/api/auth/lichess/exchange")
def exchange(req: Exchange, request: Request):
    with request.app.state.sessions() as sess:
        lc = sess.get(LoginCode, hash_token(req.code))
        if lc is None or lc.created_at < time.time() - CODE_S:
            raise HTTPException(400, "unknown or expired login code")
        lid, uname = lc.lichess_id, lc.lichess_username
        cur = current_user(sess, request)
        if cur is not None and cur.lichess_id not in (None, lid):
            raise HTTPException(409, "this device is signed in to another Lichess account; sign out first")
        sess.delete(lc)
        known = sess.scalar(select(User).where(User.lichess_id == lid))
        merged = 0
        if known is None:
            target = cur if cur is not None else User(token_hash=hash_token(secrets.token_urlsafe(32)))
            target.lichess_id = lid
            sess.add(target)
        else:
            target = known
            if cur is not None and cur.id != known.id:   # anonymous account folds into the Lichess one
                merged = sess.execute(update(Game).where(Game.user_id == cur.id).values(user_id=known.id)).rowcount
                sess.execute(delete(Rating).where(Rating.subject == f"user:{cur.id}"))
                sess.execute(delete(Consent).where(Consent.user_id == cur.id))
                sess.execute(delete(AuthToken).where(AuthToken.user_id == cur.id))
                sess.delete(cur)
        target.lichess_username = uname
        if not target.display_name:
            target.display_name = uname
        sess.flush()
        token = secrets.token_urlsafe(32)
        sess.add(AuthToken(token_hash=hash_token(token), user_id=target.id))
        sess.commit()
        return {"token": token, "merged_games": merged, "user": me(sess, target)}
