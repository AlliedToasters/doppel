"""Consented import of a user's public Lichess game history (consent history-v1).

    PUT  /api/me/history-consent {granted, version, adult}   grant (18+ box ticked) or withdraw; withdrawing
                                                             deletes the imported games and the import status
    POST /api/me/history/import                              queue an import (202); returns the status
    GET  /api/me/history                                     the status, polled by the account page

Who may import: an account signed in with Lichess, not in Lichess kid mode (read at sign-in through
GET /api/account/kid, lichess_login.py), holding history-v1 with the self-declared 18+ flag.

The import uses Lichess's public export, GET /api/games/user/{id} as ndjson, with no token: the backend never
holds a Lichess token. It follows Lichess's API rules: one export stream at a time across the whole server (a
global lock; later imports wait as "queued"), and after a 429 nothing is sent to Lichess for at least 60 s.
It runs as a background task in the api process, newest games first, at most DOPPEL_HISTORY_MAX_GAMES per run.
Re-running is incremental: `since` is the newest stored game's createdAt as of the last completed run (an
interrupted run is fetched again from that point; games already stored are skipped). A run that hits the cap
fetches the newest games only, so a gap of older-than-cap games since the previous run is left unfetched.

Training is not here: the rows are kept for the personal doppel (and for general models only alongside the
training consent, once the export learns about them).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select

from .accounts import HISTORY_VERSION, _me_or_401, holds_consent, me
from .lichess_login import USER_AGENT
from .db import Consent, HistoryImport, ImportedGame, User, delete_history

log = logging.getLogger("doppel_api.history")
router = APIRouter()

PERF_TYPES = "ultraBullet,bullet,blitz,rapid,classical,correspondence"   # standard chess only, no variants
BACKOFF_S = 60      # Lichess: after a 429, wait at least a minute
ATTEMPTS = 3
BATCH = 100
_UNFINISHED = {"created", "started", "aborted", "noStart", "unknownFinish"}
_sleep = asyncio.sleep   # tests replace this


def max_games() -> int:
    return max(1, int(os.environ.get("DOPPEL_HISTORY_MAX_GAMES", "5000")))


class ImportFailed(Exception):
    """Ends a run with state "error" and this message (shown to the user)."""


class _Withdrawn(Exception):
    """The consent or the account went away mid-run: stop without writing anything."""


def parse(g: dict, lichess_id: str) -> tuple[int, dict] | None:
    """One ndjson game -> (createdAt ms, ImportedGame fields), or None for games not worth keeping
    (variants, from-position, unfinished, not the user's)."""
    if g.get("variant", "standard") != "standard" or g.get("initialFen") or g.get("status") in _UNFINISHED:
        return None
    players = g.get("players") or {}
    color = next((c for c in ("white", "black")
                  if ((players.get(c) or {}).get("user") or {}).get("id") == lichess_id), None)
    moves = (g.get("moves") or "").split()
    if color is None or not moves:
        return None
    opp = "black" if color == "white" else "white"
    winner = g.get("winner")
    result = {"white": "1-0", "black": "0-1"}.get(winner, "1/2-1/2")
    clock, opening = g.get("clock") or {}, g.get("opening") or {}
    created = int(g["createdAt"])
    return created, {
        "lichess_game_id": str(g["id"])[:16], "played_at": created / 1000,
        "speed": str(g.get("speed", ""))[:16], "perf": str(g.get("perf", ""))[:16], "rated": bool(g.get("rated")),
        "color": color, "rating": players[color].get("rating"), "opp_rating": (players.get(opp) or {}).get("rating"),
        "result": result, "status": str(g.get("status", ""))[:24],
        "base_s": clock.get("initial"), "inc_s": clock.get("increment"),
        "moves": moves, "clocks": g.get("clocks"),
        "opening_eco": (opening.get("eco") or None) and opening["eco"][:8],
        "opening_name": (opening.get("name") or None) and opening["name"][:128],
    }


class Importer:
    """Runs imports one at a time (the lock) and remembers the 429 backoff. One per app, made in the lifespan."""

    def __init__(self, app):
        self.app = app
        self.lock = asyncio.Lock()
        self.not_before = 0.0   # epoch s: no request to Lichess before this
        self.tasks: dict[str, asyncio.Task] = {}

    def recover(self) -> None:
        """Runs at startup: a run cut short by a restart is reported, not silently stuck at "running"."""
        with self.app.state.sessions() as sess:
            for row in sess.scalars(select(HistoryImport).where(HistoryImport.state.in_(("queued", "running")))):
                row.state, row.error, row.updated_at = "error", "Interrupted by a server restart. Import again.", time.time()
            sess.commit()

    def start(self, user_id: str) -> None:
        t = self.tasks.get(user_id)
        if t is not None and not t.done():
            return
        t = asyncio.create_task(self._run(user_id))
        self.tasks[user_id] = t
        t.add_done_callback(lambda done: self.tasks.pop(user_id, None) if self.tasks.get(user_id) is done else None)

    def cancel(self, user_id: str) -> None:
        t = self.tasks.pop(user_id, None)
        if t is not None:
            t.cancel()

    async def close(self) -> None:
        ts = list(self.tasks.values())
        for t in ts:
            t.cancel()
        await asyncio.gather(*ts, return_exceptions=True)

    def _update(self, user_id: str, **fields) -> None:
        """Change the status row if it still exists (a withdrawal deletes it; never recreate it)."""
        with self.app.state.sessions() as sess:
            row = sess.get(HistoryImport, user_id)
            if row is not None:
                for k, v in fields.items():
                    setattr(row, k, v)
                row.updated_at = time.time()
                sess.commit()

    async def _run(self, user_id: str) -> None:
        try:
            async with self.lock:
                self._update(user_id, state="running")
                await self._import(user_id)
        except _Withdrawn:
            pass
        except ImportFailed as e:
            self._update(user_id, state="error", error=str(e)[:200])
        except Exception:
            log.exception("history import failed")
            self._update(user_id, state="error", error="The import failed. Try again later.")

    async def _wait_backoff(self) -> None:
        delay = self.not_before - time.time()
        if delay > 0:
            await _sleep(delay)

    async def _import(self, user_id: str) -> None:
        with self.app.state.sessions() as sess:
            u, row = sess.get(User, user_id), sess.get(HistoryImport, user_id)
            if u is None or row is None or not u.lichess_id:
                raise _Withdrawn
            lid, since = u.lichess_id, row.cursor_ms
        params = {"max": max_games(), "perfType": PERF_TYPES, "moves": "true", "clocks": "true", "opening": "true",
                  "pgnInJson": "false", "finished": "true", "sort": "dateDesc"}
        if since:
            params["since"] = since
        lc: httpx.AsyncClient = self.app.state.lichess
        for _ in range(ATTEMPTS):
            await self._wait_backoff()
            async with lc.stream("GET", f"/api/games/user/{lid}", params=params,
                                 headers={"accept": "application/x-ndjson", "user-agent": USER_AGENT},
                                 timeout=httpx.Timeout(10, read=120)) as r:
                if r.status_code == 429:
                    try:
                        after = float(r.headers.get("retry-after", 0))
                    except ValueError:
                        after = 0
                    self.not_before = time.time() + max(BACKOFF_S, after)
                    continue
                if r.status_code == 404:
                    raise ImportFailed("Lichess has no games for this account (closed or renamed?).")
                if r.status_code != 200:
                    raise ImportFailed(f"Lichess answered {r.status_code}. Try again later.")
                newest, batch, added = since or 0, [], 0
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    p = parse(json.loads(line), lid)
                    if p is None:
                        continue
                    newest = max(newest, p[0])
                    batch.append(p[1])
                    if len(batch) >= BATCH:
                        added += self._store(user_id, batch, added)
                        batch = []
                added += self._store(user_id, batch, added)
            self._update(user_id, state="done", error=None, cursor_ms=newest or None)
            return
        raise ImportFailed("Lichess is busy (rate limit). Try again in a few minutes.")

    def _store(self, user_id: str, rows: list[dict], added_so_far: int) -> int:
        """Write a batch, skipping games already stored. Synchronous and on the event loop, like the consent
        withdrawal and DELETE /api/me, so a withdrawal can never land between the consent check and the write."""
        with self.app.state.sessions() as sess:
            if sess.get(HistoryImport, user_id) is None or holds_consent(sess, user_id, "history") is None:
                raise _Withdrawn
            ids = {r["lichess_game_id"] for r in rows}
            have = set(sess.scalars(select(ImportedGame.lichess_game_id).where(
                ImportedGame.user_id == user_id, ImportedGame.lichess_game_id.in_(ids)))) if ids else set()
            new = {r["lichess_game_id"]: r for r in rows if r["lichess_game_id"] not in have}
            for r in new.values():
                sess.add(ImportedGame(user_id=user_id, **r))
            row = sess.get(HistoryImport, user_id)
            row.added, row.updated_at = added_so_far + len(new), time.time()
            sess.commit()
            return len(new)


def refusal(sess, u: User) -> str | None:
    """Why this account may not import, or None."""
    if not u.lichess_id:
        return "Sign in with Lichess first."
    if u.lichess_kid:
        return "Lichess kid-mode accounts can't import their games."
    if u.lichess_kid is None:
        return "Sign in with Lichess again, so Doppel can check your account isn't in kid mode."
    c = holds_consent(sess, u.id, "history")
    if c is None:
        return "Turn on importing your Lichess games first."
    if c.version != HISTORY_VERSION:
        return "The import terms changed. Read them and turn importing on again."
    if not c.adult:
        return "You must be 18 or older to import your games."
    return None


def status(sess, u: User) -> dict:
    row = sess.get(HistoryImport, u.id)
    n = sess.scalar(select(func.count()).select_from(ImportedGame).where(ImportedGame.user_id == u.id))
    why = refusal(sess, u)
    return {"state": row.state if row else "none", "games": n, "added": row.added if row else 0,
            "error": row.error if row else None, "started_at": row.started_at if row else None,
            "updated_at": row.updated_at if row else None, "max_games": max_games(),
            "can_import": why is None, "reason": why}


class HistoryConsentIntent(BaseModel):
    granted: bool
    version: str = HISTORY_VERSION
    adult: bool = False   # "I am 18 or older", ticked by the user


@router.get("/api/me/history")
def get_history(request: Request):
    with request.app.state.sessions() as sess:
        return status(sess, _me_or_401(sess, request))


@router.post("/api/me/history/import", status_code=202)
async def post_import(request: Request):
    with request.app.state.sessions() as sess:
        u = _me_or_401(sess, request)
        why = refusal(sess, u)
        if why:
            raise HTTPException(403, why)
        row = sess.get(HistoryImport, u.id)
        if row is None:
            row = HistoryImport(user_id=u.id, state="queued")
            sess.add(row)
        elif row.state not in ("queued", "running"):
            row.state, row.added, row.error = "queued", 0, None
            row.started_at = row.updated_at = time.time()
        sess.commit()
        request.app.state.history.start(u.id)
        return status(sess, u)


@router.put("/api/me/history-consent")
async def put_history_consent(req: HistoryConsentIntent, request: Request):
    """Async on purpose: see Importer._store."""
    with request.app.state.sessions() as sess:
        u = _me_or_401(sess, request)
        held = holds_consent(sess, u.id, "history")
        if req.granted:
            if not u.lichess_id:
                raise HTTPException(403, "Sign in with Lichess first.")
            if u.lichess_kid:
                raise HTTPException(403, "Lichess kid-mode accounts can't import their games.")
            if req.version != HISTORY_VERSION:
                raise HTTPException(409, f"consent text changed: show version {HISTORY_VERSION}")
            if not req.adult:
                raise HTTPException(422, "You must be 18 or older to import your games.")
            if held is None or held.version != req.version or not held.adult:
                if held is not None:
                    held.withdrawn_at = time.time()
                sess.add(Consent(user_id=u.id, kind="history", version=req.version, adult=True))
        else:
            if held is not None:
                held.withdrawn_at = time.time()
            request.app.state.history.cancel(u.id)
            delete_history(sess, u.id)
        sess.commit()
        return me(sess, u)
