"""Doppel demo server: play the baseline bots (later Doppel itself) from a phone.

CPU-only by design (the GPU belongs to the warden queue). Stateless API: the client sends the move
list and clocks; the server replies with the bot's move, its human-like think time when the bot has
a clock model, and the bot's top move probabilities ("what players at this rating play here").

Run: CUDA_VISIBLE_DEVICES="" .venv/bin/uvicorn demo.server:app --host 127.0.0.1 --port 8612
"""

from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from doppel_lab.engines import CM_MODELS, ChessMimic, GameState, Maia3  # noqa: E402
from doppel_lab.top1 import logprobs  # noqa: E402

torch.set_num_threads(int(os.environ.get("DEMO_TORCH_THREADS", "4")))
STATIC = Path(__file__).parent / "static"
CM_BANDS = sorted(int(p.name.split("_")[0]) for p in (CM_MODELS / "move_model").iterdir()
                  if (p / "model.ckpt").stat().st_size > 1_000_000
                  and (CM_MODELS / "clock_model" / p.name / "model.ckpt").exists()
                  and (CM_MODELS / "clock_model" / p.name / "model.ckpt").stat().st_size > 1_000_000)

BOTS = {
    "maia3-79m": {"name": "Maia-3 79M", "blurb": "Most accurate per-move human model", "clock": False},
    "maia3-23m": {"name": "Maia-3 23M", "blurb": "Smaller Maia-3", "clock": False},
    "maia3-5m": {"name": "Maia-3 5M", "blurb": "Tiny Maia-3", "clock": False},
    "chessmimic": {"name": "ChessMimic", "blurb": "Per-band models with human think times", "clock": True},
}

_engines: dict[str, object] = {}
_lock = threading.Lock()
_rng = np.random.default_rng()


def engine(bot: str, elo: int):
    if bot.startswith("maia3-"):
        key = bot
        if key not in _engines:
            _engines[key] = Maia3(bot.removeprefix("maia3-"), "cpu")
        return _engines[key]
    band = min(CM_BANDS, key=lambda b: abs(b + 50 - elo))
    key = f"chessmimic-{band}"
    if key not in _engines:
        _engines[key] = ChessMimic(band, "cpu")
    return _engines[key]


class MoveReq(BaseModel):
    bot: str
    elo: int = Field(1500, ge=600, le=2800)
    moves: list[str] = []
    white_clock: float = 180.0
    black_clock: float = 180.0
    base: float = 180.0
    inc: float = 0.0
    temperature: float = Field(1.0, ge=0.0, le=2.0)


app = FastAPI(title="Doppel demo")


@app.on_event("startup")
def _warm():
    """Load every bot in the background so the first move of a demo is instant."""
    def go():
        for bot in ("maia3-79m", "chessmimic", "maia3-23m", "maia3-5m"):
            for elo in ([b + 50 for b in CM_BANDS] if bot == "chessmimic" else [1500]):
                with _lock:
                    engine(bot, elo)
    threading.Thread(target=go, daemon=True).start()


@app.get("/api/bots")
def bots():
    return {"bots": [{"id": k, **v} for k, v in BOTS.items()], "chessmimic_bands": [b + 50 for b in CM_BANDS]}


@app.post("/api/move")
def move(req: MoveReq):
    if req.bot not in BOTS:
        raise HTTPException(400, f"unknown bot {req.bot}")
    board = chess.Board()
    hist = [Maia3.board_tokens(board)]
    try:
        for m in req.moves:
            board.push_uci(m)
            hist.append(Maia3.board_tokens(board))
    except ValueError:
        raise HTTPException(400, "illegal move list")
    if board.is_game_over(claim_draw=True):
        raise HTTPException(409, "game is over")
    st = GameState(board, req.elo, req.elo, req.base, req.inc, max(req.white_clock, 0.1),
                   max(req.black_clock, 0.1), list(req.moves), hist[-8:])

    t0 = time.perf_counter()
    with _lock:
        eng = engine(req.bot, req.elo)
        lp = logprobs(eng, [st])[0]
        think = None
        if isinstance(eng, ChessMimic):
            mv, think = eng.sample([st], _rng)[0]
            if req.temperature == 0:
                mv = max(lp, key=lp.get)
        else:
            moves, logp = zip(*lp.items())
            if req.temperature == 0:
                mv = moves[int(np.argmax(logp))]
            else:
                p = np.exp(np.array(logp) / req.temperature)
                mv = moves[_rng.choice(len(moves), p=p / p.sum())]
    ms = (time.perf_counter() - t0) * 1000
    top = sorted(lp.items(), key=lambda kv: -kv[1])[:5]
    return {
        "move": mv,
        "san": board.san(chess.Move.from_uci(mv)),
        "think_s": think,
        "compute_ms": round(ms, 1),
        "top": [{"uci": u, "san": board.san(chess.Move.from_uci(u)), "p": round(math.exp(v), 4)} for u, v in top],
    }


@app.get("/api/health")
def health():
    return {"ok": True, "loaded": sorted(_engines)}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=STATIC), name="static")
