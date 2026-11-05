"""Doppel API skeleton: server-authoritative Doppel-hosted games.

The server holds the moves, the clocks and the doppel's per-game z; the client sends intents (REST or
WebSocket) and renders the state it gets back. Bot moves come from backend/infer over HTTP.
Accounts, consent, ratings and export come later (Stage 0 is tailnet-only; see ARCHITECTURE.md).

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

from . import rules
from .db import Game, make_sessionmaker


class Infer:
    """Thin client for backend/infer."""

    def __init__(self, http: httpx.AsyncClient):
        self.http = http

    async def bots(self) -> dict[str, dict]:
        r = await self.http.get("/api/bots")
        r.raise_for_status()
        return {b["id"]: b for b in r.json()["bots"]}

    async def move(self, g: Game) -> str:
        r = await self.http.post("/api/move", json={
            "bot": g.bot, "elo": g.bot_elo, "opp_elo": g.human_elo, "moves": g.moves, "z": g.z})
        r.raise_for_status()
        return r.json()["move"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "sessions"):
        app.state.sessions = make_sessionmaker()
    if not hasattr(app.state, "infer"):
        app.state.infer = Infer(httpx.AsyncClient(base_url=os.environ.get("DOPPEL_INFER_URL", "http://127.0.0.1:8614"),
                                                  timeout=10))
    yield


app = FastAPI(title="Doppel API", lifespan=lifespan)
_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_sockets: dict[str, set[WebSocket]] = defaultdict(set)
_rng = np.random.default_rng()


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
        "status": g.status, "result": g.result, "termination": g.termination,
    }


def _load(sess, game_id: str) -> Game:
    g = sess.get(Game, game_id)
    if g is None:
        raise HTTPException(404, "no such game")
    if rules.settle(g, time.time()):
        sess.commit()
    return g


async def _broadcast(g: Game) -> None:
    msg = {"type": "state", "game": state(g, time.time())}
    for ws in list(_sockets[g.id]):
        try:
            await ws.send_json(msg)
        except Exception:
            _sockets[g.id].discard(ws)


async def _bot_turn(app_: FastAPI, game_id: str) -> None:
    """If it is the bot's turn, ask infer for a move and apply it. Caller holds the game lock."""
    with app_.state.sessions() as sess:
        g = _load(sess, game_id)
        if g.status != "active" or rules.side_to_move(g) == g.human_color:
            return
        try:
            uci = await app_.state.infer.move(g)
        except httpx.HTTPError as e:
            raise HTTPException(503, f"infer unavailable: {e}")
        rules.play(g, uci, time.time())
        sess.commit()
        await _broadcast(g)


# ---------- REST ----------
class NewGame(BaseModel):
    bot: str = "maia3-79m"
    bot_elo: int = Field(1500, ge=600, le=2800)
    human_elo: int = Field(1500, ge=600, le=2800)
    color: Literal["white", "black", "random"] = "white"
    base_s: int = Field(600, ge=15, le=10800)
    inc_s: int = Field(0, ge=0, le=180)


class MoveIntent(BaseModel):
    uci: str
    ply: int | None = None  # the ply the client believes it is moving at; stale or duplicate sends are refused


@app.post("/api/games")
async def create_game(req: NewGame, request: Request):
    app_ = request.app
    bots = await app_.state.infer.bots()
    if req.bot not in bots:
        raise HTTPException(400, f"unknown bot {req.bot}")
    color = random.choice(["white", "black"]) if req.color == "random" else req.color
    g = Game(bot=req.bot, package_id=bots[req.bot]["package"], bot_elo=req.bot_elo, human_elo=req.human_elo,
             human_color=color, base_ms=req.base_s * 1000, inc_ms=req.inc_s * 1000, white_ms=req.base_s * 1000,
             black_ms=req.base_s * 1000, moves=[], z=rules.sample_z(bots[req.bot].get("z_prior"), _rng))
    with app_.state.sessions() as sess:
        sess.add(g)
        sess.commit()
    async with _locks[g.id]:
        await _bot_turn(app_, g.id)
    with app_.state.sessions() as sess:
        return state(_load(sess, g.id), time.time())


@app.get("/api/games/{game_id}")
def get_game(game_id: str, request: Request):
    with request.app.state.sessions() as sess:
        return state(_load(sess, game_id), time.time())


async def _human_move(app_: FastAPI, game_id: str, intent: MoveIntent) -> None:
    with app_.state.sessions() as sess:
        g = _load(sess, game_id)
        if g.status == "active" and rules.side_to_move(g) != g.human_color:
            raise HTTPException(409, "not your turn")
        if intent.ply is not None and intent.ply != len(g.moves):
            raise HTTPException(409, f"stale move: game is at ply {len(g.moves)}")
        try:
            rules.play(g, intent.uci, time.time())
        except rules.IllegalMove as e:
            raise HTTPException(400, str(e))
        sess.commit()
        await _broadcast(g)


@app.post("/api/games/{game_id}/move")
async def post_move(game_id: str, intent: MoveIntent, request: Request):
    """Apply the human's move, then the bot's reply; returns the state after both."""
    async with _locks[game_id]:
        await _human_move(request.app, game_id, intent)
        await _bot_turn(request.app, game_id)
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
    async with _locks[game_id]:
        await _resign(request.app, game_id)
    return get_game(game_id, request)


# ---------- WebSocket ----------
@app.websocket("/api/games/{game_id}/ws")
async def game_ws(ws: WebSocket, game_id: str):
    """Intents in: {"type":"move","uci":..,"ply":..} | {"type":"resign"} | {"type":"sync"}.
    Out: {"type":"state","game":{..}} after every change (the human's move, then the bot's), or
    {"type":"error","status":..,"detail":..}."""
    app_ = ws.app
    with app_.state.sessions() as sess:
        g = sess.get(Game, game_id)
        if g is None:
            await ws.close(code=4404)
            return
    await ws.accept()
    _sockets[game_id].add(ws)
    try:
        with app_.state.sessions() as sess:
            await ws.send_json({"type": "state", "game": state(_load(sess, game_id), time.time())})
        while True:
            msg = await ws.receive_json()
            try:
                async with _locks[game_id]:
                    if msg.get("type") == "move":
                        await _human_move(app_, game_id, MoveIntent(uci=msg["uci"], ply=msg.get("ply")))
                        await _bot_turn(app_, game_id)
                    elif msg.get("type") == "resign":
                        await _resign(app_, game_id)
                    else:
                        with app_.state.sessions() as sess:
                            await ws.send_json({"type": "state", "game": state(_load(sess, game_id), time.time())})
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


@app.post("/api/move")
async def stateless_move(request: Request):
    """Stateless analysis/practice move (the pre-api client contract). Not used for hosted games."""
    r = await request.app.state.infer.http.post("/api/move", content=await request.body(),
                                                headers={"content-type": "application/json"})
    return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/api/health")
def health():
    return {"ok": True}


if os.environ.get("DOPPEL_STATIC"):
    app.mount("/", StaticFiles(directory=os.environ["DOPPEL_STATIC"], html=True), name="static")
