"""Stateless CPU move server: model packages in, moves out. Replaces demo/server.py.

The client sends the whole move list each turn; nothing is stored. Bots are model packages found under
DOPPEL_MODELS (default ./models); a bot id is the package's manifest name (e.g. "maia3-79m"); when several
packages share a name, the newest directory wins.

Run (repo root): .venv/bin/uvicorn doppel_infer.app:app --host 127.0.0.1 --port 8612
Env: DOPPEL_MODELS, DOPPEL_THREADS (ORT intra-op threads per package, default 4),
     DOPPEL_STATIC (optional dir served at / — Stage 0 uses it for the APK download).
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import chess
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from doppel_model import ModelPackage

MODELS = Path(os.environ.get("DOPPEL_MODELS", "models"))
THREADS = int(os.environ.get("DOPPEL_THREADS", "4"))
BLURBS = {
    "maia3-79m": "Most accurate per-move human model",
    "maia3-23m": "Smaller Maia-3",
    "maia3-5m": "Tiny Maia-3",
}
ORDER = ["maia3-79m", "maia3-23m", "maia3-5m"]


def _load() -> dict[str, ModelPackage]:
    by_name: dict[str, ModelPackage] = {}
    dirs = sorted((p for p in MODELS.iterdir() if (p / "manifest.json").exists()), key=lambda p: p.stat().st_mtime)
    for d in dirs:
        pkg = ModelPackage.load(d, threads=THREADS)
        by_name[pkg.manifest["name"]] = pkg
    return by_name


PACKAGES = _load() if MODELS.exists() else {}
_rng = np.random.default_rng()
app = FastAPI(title="Doppel infer")


class MoveReq(BaseModel):
    bot: str
    elo: int = Field(1500, ge=600, le=2800)
    opp_elo: int | None = Field(None, ge=600, le=2800)
    moves: list[str] = []
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    z: list[float] | None = None  # per-game z from the api; ignored by stateless packages (z_prior null)


def _display(m: dict) -> str:
    b = m.get("base_model", {})
    return f"Maia-3 {b['size'].upper()}" if b.get("family") == "maia3" else m["name"]


@app.get("/api/bots")
def bots():
    names = sorted(PACKAGES, key=lambda n: ORDER.index(n) if n in ORDER else len(ORDER))
    return {"bots": [{"id": n, "name": _display(PACKAGES[n].manifest),
                      "blurb": BLURBS.get(n, ""), "clock": PACKAGES[n].manifest.get("clock_head", False),
                      "z_prior": PACKAGES[n].manifest.get("z_prior"),
                      "package": PACKAGES[n].id} for n in names]}


@app.post("/api/move")
def move(req: MoveReq):
    pkg = PACKAGES.get(req.bot)
    if pkg is None:
        raise HTTPException(400, f"unknown bot {req.bot}")
    t0 = time.perf_counter()
    try:
        board, legal, lp = pkg.policy(req.moves, req.elo, req.opp_elo or req.elo)
    except ValueError:
        raise HTTPException(400, "illegal move list")
    if board.is_game_over(claim_draw=True):
        raise HTTPException(409, "game is over")
    if req.temperature == 0:
        i = int(np.argmax(lp))
    else:
        p = np.exp(lp / req.temperature)
        i = int(_rng.choice(len(legal), p=p / p.sum()))
    mv = legal[i]
    ms = (time.perf_counter() - t0) * 1000
    top = np.argsort(-lp)[:5]
    return {
        "move": mv,
        "san": board.san(chess.Move.from_uci(mv)),
        "think_s": None,
        "compute_ms": round(ms, 1),
        "package": pkg.id,
        "top": [{"uci": legal[j], "san": board.san(chess.Move.from_uci(legal[j])), "p": round(math.exp(lp[j]), 4)}
                for j in top],
    }


@app.get("/api/health")
def health():
    return {"ok": True, "packages": {n: p.id for n, p in PACKAGES.items()}}


if os.environ.get("DOPPEL_STATIC"):
    app.mount("/", StaticFiles(directory=os.environ["DOPPEL_STATIC"], html=True), name="static")
