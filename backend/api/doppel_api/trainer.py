"""The trainer's side of the api: bigbox pulls doppel jobs and pushes adapters back (it is never reached from here).

    POST /api/trainer/claim                      the oldest queued doppel as a job, or {"job": null}
    POST /api/trainer/jobs/{job}/adapter         body: the adapter file (doppel_model.adapter), <= 1 MB
    POST /api/trainer/jobs/{job}/fail {error}    the job failed; the user sees a plain "try again"

Auth: Authorization: Bearer <trainer token>; the api keeps only its sha256 (DOPPEL_TRAINER_TOKEN_SHA256) and
answers 404 on every trainer path when that is unset. A job carries a random id, the format-2 package name and
the games (SAN moves, the user's colour and both ratings): no user id, no Lichess id, no game ids.

Claiming: a doppel "training" for longer than TIMEOUT_S becomes "error" (the user can retry); at most
DOPPEL_TRAIN_DAILY (default 200) claims per rolling 24 h; beyond that queued doppels wait.
Results: the adapter is checked against the package's lora spec (doppel_model.adapter.read_adapter: no pickle,
names, shapes, float16, finite, made for this package), written as <id>.npz under DOPPEL_ADAPTERS, and the doppel
becomes "ready". A job whose doppel was deleted or re-requested meanwhile answers 410 and nothing is kept.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from doppel_model.adapter import read_adapter

from .db import Doppel, ImportedGame

log = logging.getLogger("doppel_api.trainer")
router = APIRouter()

TIMEOUT_S = 1800
MAX_BYTES = 1 << 20
JOB = re.compile(r"^[0-9a-f]{32}$")


def package_name() -> str:
    return os.environ.get("DOPPEL_DOPPEL_PACKAGE", "maia3-79m-lora8")


def daily_cap() -> int:
    return int(os.environ.get("DOPPEL_TRAIN_DAILY", "200"))


def adapters_dir() -> Path:
    return Path(os.environ.get("DOPPEL_ADAPTERS", "adapters"))


def lora_spec(name: str) -> dict:
    """The lora block of the newest format-2 package with this name under DOPPEL_MODELS."""
    root = Path(os.environ.get("DOPPEL_MODELS", "models"))
    mans = sorted((p for p in root.glob(f"{name}-*") if (p / "manifest.json").exists()), key=lambda p: p.stat().st_mtime)
    for p in reversed(mans):
        m = json.loads((p / "manifest.json").read_text())
        if m.get("name") == name and m.get("lora"):
            return m["lora"]
    raise HTTPException(503, f"no format-2 package {name} on this server")


def _auth(request: Request) -> None:
    want = os.environ.get("DOPPEL_TRAINER_TOKEN_SHA256", "")
    if not want:
        raise HTTPException(404, "Not Found")
    got = request.headers.get("authorization", "")
    got = got[7:] if got.lower().startswith("bearer ") else ""
    if not got or not hmac.compare_digest(hashlib.sha256(got.encode()).hexdigest(), want.lower()):
        raise HTTPException(401, "bad trainer token")


def _job_row(sess, job: str) -> Doppel:
    if not JOB.match(job):
        raise HTTPException(404, "unknown job")
    d = sess.scalar(select(Doppel).where(Doppel.job_id == job))
    if d is None or d.state != "training":
        raise HTTPException(410, "job gone (doppel deleted, re-requested or timed out)")
    return d


@router.post("/api/trainer/claim")
def claim(request: Request):
    _auth(request)
    now = time.time()
    with request.app.state.sessions() as sess:
        for d in sess.scalars(select(Doppel).where(Doppel.state == "training", Doppel.claimed_at < now - TIMEOUT_S)):
            d.state, d.error, d.job_id, d.updated_at = "error", "Training timed out. Try again.", None, now
        sess.commit()
        n = sess.scalar(select(func.count()).select_from(Doppel).where(Doppel.claimed_at > now - 86400))
        if n >= daily_cap():
            return {"job": None, "reason": "daily cap"}
        d = sess.scalar(select(Doppel).where(Doppel.state == "queued").order_by(Doppel.requested_at).limit(1))
        if d is None:
            return {"job": None}
        name = package_name()
        lora_spec(name)                                   # 503 before claiming if the package is missing
        rows = sess.scalars(select(ImportedGame).where(ImportedGame.user_id == d.user_id,
                                                       ImportedGame.lichess_game_id.in_(d.games))
                            .order_by(ImportedGame.played_at)).all()
        d.state, d.job_id, d.claimed_at, d.updated_at = "training", secrets.token_hex(16), now, now
        sess.commit()
        return {"job": d.job_id, "package": name, "speed": d.speed,
                "games": [{"moves": g.moves, "color": g.color, "rating": g.rating, "opp_rating": g.opp_rating}
                          for g in rows]}


@router.post("/api/trainer/jobs/{job}/adapter")
async def adapter(job: str, request: Request):
    _auth(request)
    if int(request.headers.get("content-length") or 0) > MAX_BYTES:
        raise HTTPException(413, "adapter too large")
    data = await request.body()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "adapter too large")
    name = package_name()
    try:
        aid, _, meta = read_adapter(data, lora_spec(name))
    except ValueError as e:
        raise HTTPException(422, f"bad adapter: {e}")
    if meta.get("package") != name:
        raise HTTPException(422, f"adapter made for {meta.get('package')}, not {name}")
    with request.app.state.sessions() as sess:
        d = _job_row(sess, job)
        dirp = adapters_dir()
        dirp.mkdir(parents=True, exist_ok=True)
        tmp = dirp / f".{aid}.{job}.tmp"
        tmp.write_bytes(data)
        tmp.replace(dirp / f"{aid}.npz")
        old = d.adapter_id
        d.state, d.adapter_id, d.package, d.error, d.job_id, d.updated_at = "ready", aid, name, None, None, time.time()
        sess.commit()
        if old and old != aid and sess.scalar(select(Doppel.user_id).where(Doppel.adapter_id == old)) is None:
            (dirp / f"{old}.npz").unlink(missing_ok=True)
        return {"ok": True, "adapter": aid}


class Failure(BaseModel):
    error: str = Field("", max_length=500)


@router.post("/api/trainer/jobs/{job}/fail")
def fail(job: str, f: Failure, request: Request):
    _auth(request)
    with request.app.state.sessions() as sess:
        d = _job_row(sess, job)
        log.warning("doppel training failed (job %s): %s", job, f.error)
        d.state, d.error, d.job_id, d.updated_at = "error", "Training failed. Try again.", None, time.time()
        sess.commit()
        return {"ok": True}
