"""The user's personal doppel: which imported games it trains on, and its training state.

    GET  /api/me/doppel             status: eligible games per speed, what's needed, the doppel's state
    POST /api/me/doppel {speed?}    request training (202): state "queued"; the trainer (step 4) takes it from there

Eligible games: imported (history.py, consent history-v1), rated, against a person (opp_bot False: not a Lichess
BOT and not the Lichess AI, which is never rated), at one speed among bullet, blitz, rapid and classical. A doppel
trains on the most recent TARGET (200) of them, the lab recipe (195 train + 5 validation, allie_ttt --ctx 200),
and needs at least DOPPEL_MIN_GAMES (default 40; the lab's 40-game adapter gained +0.80 top-1). With no speed
given, the speed with the most eligible games.

Limits: one request per account per WEEK_S unless the last one failed; never while one is queued or training. The
Lichess account is checked again at request time (history.Importer.check_account: not flagged, closed or a BOT).
The previous adapter stays playable while a new one trains. Withdrawing history-v1 or deleting the account deletes
the doppel and its adapter file (db.delete_history).
"""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select

from .accounts import _me_or_401
from .db import Doppel, HistoryImport, ImportedGame, User
from .history import ImportFailed, refusal as history_refusal

router = APIRouter()

SPEEDS = ("bullet", "blitz", "rapid", "classical")
TARGET = 200
WEEK_S = 7 * 86400


def min_games() -> int:
    return max(15, int(os.environ.get("DOPPEL_MIN_GAMES", "40")))   # below ~15 the 5-game validation split fails


def _eligible_q(user_id: str):
    return (ImportedGame.user_id == user_id, ImportedGame.rated.is_(True), ImportedGame.opp_bot.is_(False),
            ImportedGame.speed.in_(SPEEDS))


def eligible(sess, user_id: str) -> dict[str, int]:
    rows = sess.execute(select(ImportedGame.speed, func.count()).where(*_eligible_q(user_id))
                        .group_by(ImportedGame.speed)).all()
    got = dict(rows)
    return {s: int(got.get(s, 0)) for s in SPEEDS}


def pick_speed(counts: dict[str, int], speed: str | None) -> str:
    return speed if speed else max(SPEEDS, key=lambda s: (counts[s], -SPEEDS.index(s)))


def refusal(sess, u: User, speed: str | None = None, now: float | None = None) -> str | None:
    """Why this account may not request a doppel now, or None."""
    why = history_refusal(sess, u)
    if why:
        return why
    imp = sess.get(HistoryImport, u.id)
    if imp is None:
        return "Import your Lichess games first."
    if imp.state in ("queued", "running"):
        return "Wait for the import to finish."
    counts = eligible(sess, u.id)
    s = pick_speed(counts, speed)
    if counts[s] < min_games():
        return (f"A doppel needs {min_games()} rated games against people at one speed; "
                f"you have {counts[s]} at {s}.")
    d = sess.get(Doppel, u.id)
    if d is not None:
        if d.state in ("queued", "training"):
            return "Your doppel is already in training."
        now = time.time() if now is None else now
        if d.state != "error" and now - d.requested_at < WEEK_S:
            return f"One new doppel per week: next on {time.strftime('%Y-%m-%d', time.gmtime(d.requested_at + WEEK_S))}."
    return None


def status(sess, u: User) -> dict:
    d = sess.get(Doppel, u.id)
    counts = eligible(sess, u.id)
    why = refusal(sess, u)
    return {"state": d.state if d else "none", "speed": d.speed if d else None,
            "games_used": len(d.games) if d else 0, "adapter": d.adapter_id if d else None,
            "package": d.package if d else None, "error": d.error if d else None,
            "requested_at": d.requested_at if d else None, "updated_at": d.updated_at if d else None,
            "eligible": counts, "best_speed": pick_speed(counts, None), "need": min_games(), "target": TARGET,
            "can_request": why is None, "reason": why}


class DoppelRequest(BaseModel):
    speed: str | None = None


@router.get("/api/me/doppel")
def get_doppel(request: Request):
    with request.app.state.sessions() as sess:
        return status(sess, _me_or_401(sess, request))


@router.post("/api/me/doppel", status_code=202)
async def post_doppel(req: DoppelRequest, request: Request):
    if req.speed is not None and req.speed not in SPEEDS:
        raise HTTPException(422, f"speed must be one of {', '.join(SPEEDS)}")
    with request.app.state.sessions() as sess:
        u = _me_or_401(sess, request)
        why = refusal(sess, u, req.speed)
        if why:
            raise HTTPException(403, why)
        lid = u.lichess_id
    try:
        await request.app.state.history.check_account(lid)
    except ImportFailed as e:
        raise HTTPException(403, str(e))
    with request.app.state.sessions() as sess:
        u = _me_or_401(sess, request)
        why = refusal(sess, u, req.speed)          # again: the account check awaited, things may have moved
        if why:
            raise HTTPException(403, why)
        speed = pick_speed(eligible(sess, u.id), req.speed)
        ids = sess.scalars(select(ImportedGame.lichess_game_id).where(*_eligible_q(u.id), ImportedGame.speed == speed)
                           .order_by(ImportedGame.played_at.desc()).limit(TARGET)).all()[::-1]
        now = time.time()
        d = sess.get(Doppel, u.id)
        if d is None:
            d = Doppel(user_id=u.id)
            sess.add(d)
        d.state, d.speed, d.games, d.error, d.requested_at, d.updated_at = "queued", speed, list(ids), None, now, now
        sess.commit()
        return status(sess, u)
