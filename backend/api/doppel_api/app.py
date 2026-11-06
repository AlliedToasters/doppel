"""Doppel API skeleton: server-authoritative Doppel-hosted games.

The server holds the moves, the clocks and the doppel's per-game z; the client sends intents (REST or
WebSocket) and renders the state it gets back. Bot moves come from backend/infer over HTTP.
Accounts and consent: accounts.py (anonymous bearer-token accounts, versioned training consent).
Ratings: ratings.py (MAP Bradley-Terry, refit when a rated game ends). Export: export.py (consented only).

Run (repo root): see backend/api/README.md (Stage 0: port 8612, infer on 8614)
Env: DOPPEL_STATIC (optional dir served at /, e.g. demo/static for the APK), DOPPEL_DB_URL (default sqlite:///data/api.db), DOPPEL_INFER_URL (default http://127.0.0.1:8614)
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select

from . import accounts, ratings, rules
from .db import Game, Ply, make_sessionmaker


class Infer:
    """Thin client for backend/infer."""

    def __init__(self, http: httpx.AsyncClient):
        self.http = http

    async def bots(self) -> dict[str, dict]:
        r = await self.http.get("/api/bots")
        r.raise_for_status()
        return {b["id"]: b for b in r.json()["bots"]}

    async def move(self, g: Game) -> tuple[str, float]:
        """The sampled move and the policy's top-move probability (drives think time)."""
        r = await self.http.post("/api/move", json={
            "bot": g.bot, "elo": g.bot_elo, "opp_elo": g.human_elo, "moves": g.moves, "z": g.z})
        r.raise_for_status()
        d = r.json()
        return d["move"], float(d["top"][0]["p"]) if d.get("top") else 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "sessions"):
        app.state.sessions = make_sessionmaker()
    if not hasattr(app.state, "infer"):
        app.state.infer = Infer(httpx.AsyncClient(base_url=os.environ.get("DOPPEL_INFER_URL", "http://127.0.0.1:8614"),
                                                  timeout=10))
    yield


accounts.install_log_redaction()
app = FastAPI(title="Doppel API", lifespan=lifespan)
app.include_router(accounts.router)
_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_sockets: dict[str, set[WebSocket]] = defaultdict(set)
_bot_busy: set[str] = set()  # games whose bot turn is in flight
_pace_changed: dict[str, asyncio.Event] = {}  # wakes a realtime wait when the human switches to fast
_tasks: set[asyncio.Task] = set()
_rng = np.random.default_rng()
_refit_lock = asyncio.Lock()


def state(g: Game, now: float) -> dict:
    b = rules.board(g)
    return {
        "id": g.id, "bot": g.bot, "package_id": g.package_id, "bot_elo": g.bot_elo, "human_elo": g.human_elo,
        "human_color": g.human_color, "moves": g.moves, "ply": len(g.moves), "fen": b.fen(),
        "turn": rules.side_to_move(g),
        "clock": {"white_ms": max(0, rules.remaining_ms(g, "white", now)),
                  "black_ms": max(0, rules.remaining_ms(g, "black", now)),
                  "running": g.status == "active" and g.turn_started is not None,
                  "base_ms": g.base_ms, "inc_ms": g.inc_ms},
        "pace": g.bot_pace, "status": g.status, "result": g.result, "termination": g.termination,
    }


def _load(sess, game_id: str) -> Game:
    g = sess.get(Game, game_id)
    if g is None:
        raise HTTPException(404, "no such game")
    if rules.settle(g, time.time()):
        sess.commit()
    return g


def _owned(sess, game_id: str, request: Request | WebSocket) -> Game:
    """_load plus the ownership check: an account's game needs that account's token."""
    g = _load(sess, game_id)
    accounts.authorize(g, accounts.current_user(sess, request))
    return g


def _record(sess, g: Game, rec: dict, actor: str, think_source: str, client_ms: int | None = None,
            premove: bool = False) -> None:
    sess.add(Ply(game_id=g.id, ply=len(g.moves) - 1, actor=actor, think_source=think_source, pace=g.bot_pace,
                 client_ms=client_ms, premove=premove, **rec))


async def _broadcast(g: Game) -> None:
    await _send_all(g.id, {"type": "state", "game": state(g, time.time())})
    if g.status == "over" and ratings.rated(g):
        _spawn_refit()


def _spawn_refit() -> None:
    async def run():
        async with _refit_lock:
            await asyncio.to_thread(ratings.refit, app.state.sessions)
    t = asyncio.create_task(run())
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


async def _send_all(game_id: str, msg: dict) -> None:
    for ws in list(_sockets[game_id]):
        try:
            await ws.send_json(msg)
        except Exception:
            _sockets[game_id].discard(ws)


async def _bot_turn(app_: FastAPI, game_id: str) -> None:
    """If it is the bot's turn: get a move from infer, sample a think time, wait it out on the wall clock if
    the pace is realtime, then apply the move charging the think time to the bot's clock.
    Takes the game lock itself, and not while waiting, so the human can still resign or change pace."""
    if game_id in _bot_busy:
        return
    _bot_busy.add(game_id)
    try:
        async with _locks[game_id]:
            with app_.state.sessions() as sess:
                g = _load(sess, game_id)
                if g.status != "active" or rules.side_to_move(g) == g.human_color:
                    return
                ply = len(g.moves)
        t0 = time.time()
        try:
            uci, p_top = await app_.state.infer.move(g)
        except httpx.HTTPError as e:
            await _send_all(game_id, {"type": "error", "status": 503, "detail": f"infer unavailable: {e}"})
            raise HTTPException(503, f"infer unavailable: {e}")
        think = rules.think_ms(g, p_top, t0, _rng)
        wait = think / 1000 - (time.time() - t0)
        if g.bot_pace == "realtime" and wait > 0:
            ev = _pace_changed.setdefault(game_id, asyncio.Event())
            ev.clear()
            try:
                await asyncio.wait_for(ev.wait(), wait)
            except TimeoutError:
                pass
        async with _locks[game_id]:
            with app_.state.sessions() as sess:
                g = _load(sess, game_id)
                if g.status != "active" or len(g.moves) != ply:
                    return  # resigned, flagged, or already answered meanwhile
                rec = rules.play(g, uci, time.time(), charge_ms=think)
                _record(sess, g, rec, "bot", "heuristic_v1")  # synthetic timing: never a think-time label
                sess.commit()
                await _broadcast(g)
    finally:
        _bot_busy.discard(game_id)


def _spawn_bot_turn(app_: FastAPI, game_id: str) -> None:
    async def run():
        try:
            await _bot_turn(app_, game_id)
        except HTTPException:
            pass  # already reported to the sockets
    t = asyncio.create_task(run())
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


async def _set_pace(app_: FastAPI, game_id: str, pace: str) -> None:
    with app_.state.sessions() as sess:
        g = _load(sess, game_id)
        g.bot_pace = pace
        sess.commit()
        await _broadcast(g)
    if pace == "fast" and game_id in _pace_changed:
        _pace_changed[game_id].set()


# ---------- REST ----------
class NewGame(BaseModel):
    bot: str = "maia3-79m"
    bot_elo: int = Field(1500, ge=600, le=2800)
    human_elo: int = Field(1500, ge=600, le=2800)
    color: Literal["white", "black", "random"] = "white"
    base_s: int = Field(600, ge=15, le=10800)
    inc_s: int = Field(0, ge=0, le=180)
    pace: Literal["realtime", "fast"] = "realtime"
    player: str | None = Field(None, max_length=64)  # device install id (legacy; accounts use the token)


class PaceIntent(BaseModel):
    pace: Literal["realtime", "fast"]


class MoveIntent(BaseModel):
    uci: str
    ply: int | None = None  # the ply the client believes it is moving at; stale or duplicate sends are refused
    think_ms: int | None = Field(None, ge=0)  # measured on the device from the position appearing to the move
    premove: bool = False


@app.post("/api/games")
async def create_game(req: NewGame, request: Request):
    app_ = request.app
    with app_.state.sessions() as sess:
        user = accounts.current_user(sess, request)
    if user is None and accounts.require_auth():
        raise HTTPException(401, "sign in first")
    bots = await app_.state.infer.bots()
    if req.bot not in bots:
        raise HTTPException(400, f"unknown bot {req.bot}")
    color = random.choice(["white", "black"]) if req.color == "random" else req.color
    g = Game(bot=req.bot, package_id=bots[req.bot]["package"], bot_elo=req.bot_elo, human_elo=req.human_elo,
             human_color=color, base_ms=req.base_s * 1000, inc_ms=req.inc_s * 1000, white_ms=req.base_s * 1000,
             black_ms=req.base_s * 1000, moves=[], z=rules.sample_z(bots[req.bot].get("z_prior"), _rng),
             bot_pace=req.pace, player=req.player, user_id=user.id if user else None, last_move_at=time.time())
    with app_.state.sessions() as sess:
        sess.add(g)
        sess.commit()
    if req.pace == "fast":
        await _bot_turn(app_, g.id)  # reply is immediate: include the bot's first move
    else:
        _spawn_bot_turn(app_, g.id)  # arrives over the WebSocket after its think time
    with app_.state.sessions() as sess:
        return state(_load(sess, g.id), time.time())


@app.get("/api/games/{game_id}")
def get_game(game_id: str, request: Request):
    with request.app.state.sessions() as sess:
        return state(_owned(sess, game_id, request), time.time())


async def _human_move(app_: FastAPI, game_id: str, intent: MoveIntent) -> None:
    with app_.state.sessions() as sess:
        g = _load(sess, game_id)
        if g.status == "active" and rules.side_to_move(g) != g.human_color:
            raise HTTPException(409, "not your turn")
        if intent.ply is not None and intent.ply != len(g.moves):
            raise HTTPException(409, f"stale move: game is at ply {len(g.moves)}")
        try:
            rec = rules.play(g, intent.uci, time.time())
        except rules.IllegalMove as e:
            raise HTTPException(400, str(e))
        _record(sess, g, rec, "human", "human", intent.think_ms, intent.premove)
        sess.commit()
        await _broadcast(g)


def _check_owner(request: Request | WebSocket, game_id: str) -> None:
    with request.app.state.sessions() as sess:
        _owned(sess, game_id, request)


@app.get("/api/me/games")
def my_games(request: Request, limit: int = 50):
    """The account's games, newest first (summary rows; GET /api/games/{id} for the full state)."""
    with request.app.state.sessions() as sess:
        u = accounts.current_user(sess, request)
        if u is None:
            raise HTTPException(401, "sign in first")
        gs = sess.scalars(select(Game).where(Game.user_id == u.id).order_by(Game.created_at.desc())
                          .limit(min(limit, 200))).all()
        return {"games": [{"id": g.id, "created_at": g.created_at, "bot": g.bot, "bot_elo": g.bot_elo,
                           "human_color": g.human_color, "base_s": g.base_ms // 1000, "inc_s": g.inc_ms // 1000,
                           "status": g.status, "result": g.result, "termination": g.termination,
                           "plies": len(g.moves), "rated": ratings.rated(g)} for g in gs]}


@app.post("/api/games/{game_id}/move")
async def post_move(game_id: str, intent: MoveIntent, request: Request):
    """Apply the human's move, then the bot's reply; returns the state after both (in realtime pace this
    blocks for the bot's think time; the WebSocket is the non-blocking path)."""
    _check_owner(request, game_id)
    async with _locks[game_id]:
        await _human_move(request.app, game_id, intent)
    await _bot_turn(request.app, game_id)
    return get_game(game_id, request)


@app.post("/api/games/{game_id}/pace")
async def post_pace(game_id: str, intent: PaceIntent, request: Request):
    _check_owner(request, game_id)
    async with _locks[game_id]:
        await _set_pace(request.app, game_id, intent.pace)
    return get_game(game_id, request)


async def _resign(app_: FastAPI, game_id: str) -> None:
    with app_.state.sessions() as sess:
        g = _load(sess, game_id)
        try:
            rules.resign(g, g.human_color, time.time())
        except rules.IllegalMove as e:
            raise HTTPException(409, str(e))
        sess.commit()
        await _broadcast(g)


@app.post("/api/games/{game_id}/resign")
async def post_resign(game_id: str, request: Request):
    _check_owner(request, game_id)
    async with _locks[game_id]:
        await _resign(request.app, game_id)
    return get_game(game_id, request)


# ---------- WebSocket ----------
@app.websocket("/api/games/{game_id}/ws")
async def game_ws(ws: WebSocket, game_id: str):
    """Intents in: {"type":"move","uci":..,"ply":..} | {"type":"resign"} | {"type":"pace","pace":..} | {"type":"sync"}.
    Out: {"type":"state","game":{..}} after every change (the human's move, then the bot's), or
    {"type":"error","status":..,"detail":..}."""
    app_ = ws.app
    with app_.state.sessions() as sess:
        g = sess.get(Game, game_id)
        if g is None:
            await ws.close(code=4404)
            return
        try:
            accounts.authorize(g, accounts.current_user(sess, ws))
        except HTTPException as e:
            await ws.close(code=4000 + e.status_code)  # 4401 unknown token, 4403 not your game
            return
    await ws.accept()
    _sockets[game_id].add(ws)
    try:
        with app_.state.sessions() as sess:
            await ws.send_json({"type": "state", "game": state(_load(sess, game_id), time.time())})
        _spawn_bot_turn(app_, game_id)  # resume a bot turn lost to a server restart (no-op if not its turn)
        while True:
            msg = await ws.receive_json()
            try:
                async with _locks[game_id]:
                    kind = msg.get("type")
                    if kind == "move":
                        await _human_move(app_, game_id, MoveIntent(
                            uci=msg["uci"], ply=msg.get("ply"), think_ms=msg.get("think_ms"),
                            premove=bool(msg.get("premove", False))))
                    elif kind == "resign":
                        await _resign(app_, game_id)
                    elif kind == "pace":
                        await _set_pace(app_, game_id, PaceIntent(pace=msg["pace"]).pace)
                    else:
                        with app_.state.sessions() as sess:
                            await ws.send_json({"type": "state", "game": state(_load(sess, game_id), time.time())})
                _spawn_bot_turn(app_, game_id)
            except HTTPException as e:
                await ws.send_json({"type": "error", "status": e.status_code, "detail": e.detail})
    except WebSocketDisconnect:
        pass
    finally:
        _sockets[game_id].discard(ws)


# ---------- passthrough to infer (the api is the only service the client talks to) ----------
@app.get("/api/bots")
async def bots(request: Request):
    r = await request.app.state.infer.http.get("/api/bots")
    return JSONResponse(r.json(), status_code=r.status_code)


# No stateless "/api/move" here: a move for an arbitrary position is a move-suggestion service (Lichess fair play,
# ARCHITECTURE.md). Clients get bot moves only inside games the server hosts. infer stays on 127.0.0.1.


@app.get("/api/health")
def health():
    return {"ok": True}


if os.environ.get("DOPPEL_STATIC"):
    app.mount("/", StaticFiles(directory=os.environ["DOPPEL_STATIC"], html=True), name="static")
