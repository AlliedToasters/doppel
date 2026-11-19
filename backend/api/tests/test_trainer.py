"""Trainer endpoints (trainer.py): token auth, claiming (oldest first, no user ids, timeout, daily cap), adapter
upload checked against the package spec, failure, and jobs that went away."""

import hashlib
import io
import json
import time

import numpy as np
import pytest
from sqlalchemy import update
from test_doppel_api import ready
from test_history import T0, lgame
from test_lichess_login import auth, env  # noqa: F401 (env is a fixture)

from doppel_api.app import app
from doppel_api.db import Doppel
from doppel_model.adapter import save_adapter

TOKEN = "trainer-secret"
SPEC = {"inputs": [{"name": "lora_b4_linear1_A", "shape": [2, 3]}, {"name": "lora_b4_linear1_B", "shape": [4, 2]}],
        "residual_weight": 0.625}
T = {"authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def setup(env, tmp_path, monkeypatch):
    c, fake = env
    monkeypatch.setenv("DOPPEL_TRAINER_TOKEN_SHA256", hashlib.sha256(TOKEN.encode()).hexdigest())
    pkg = tmp_path / "models" / "maia3-79m-lora8-deadbeef"
    pkg.mkdir(parents=True)
    (pkg / "manifest.json").write_text(json.dumps({"name": "maia3-79m-lora8", "lora": SPEC}))
    monkeypatch.setenv("DOPPEL_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("DOPPEL_ADAPTERS", str(tmp_path / "adapters"))
    x, _ = ready(c, fake, [lgame(f"g{i:02d}", T0 + i * 1000) for i in range(45)])
    assert c.post("/api/me/doppel", json={}, headers=auth(x["token"])).status_code == 202
    return c, fake, x, tmp_path


def adapter_bytes(tmp_path, package="maia3-79m-lora8", seed=0):
    r = np.random.default_rng(seed)
    aid = save_adapter(tmp_path / "out", SPEC, [r.standard_normal((2, 3)), r.standard_normal((4, 2))],
                       {"package": package})
    return aid, (tmp_path / "out" / f"{aid}.npz").read_bytes()


def test_auth(setup, monkeypatch):
    c, *_ = setup
    assert c.post("/api/trainer/claim").status_code == 401
    assert c.post("/api/trainer/claim", headers={"authorization": "Bearer nope"}).status_code == 401
    monkeypatch.delenv("DOPPEL_TRAINER_TOKEN_SHA256")
    assert c.post("/api/trainer/claim", headers=T).status_code == 404


def test_claim_then_adapter_makes_it_ready(setup):
    c, fake, x, tmp = setup
    j = c.post("/api/trainer/claim", headers=T).json()
    assert j["package"] == "maia3-79m-lora8" and j["speed"] == "blitz" and len(j["games"]) == 45
    assert set(j["games"][0]) == {"moves", "color", "rating", "opp_rating"}           # no ids of any kind
    assert x["user"]["id"] not in json.dumps(j) and "g00" not in json.dumps(j) and "ann" not in json.dumps(j)
    assert c.post("/api/trainer/claim", headers=T).json() == {"job": None}             # nothing else queued
    assert c.get("/api/me/doppel", headers=auth(x["token"])).json()["state"] == "training"
    aid, data = adapter_bytes(tmp)
    r = c.post(f"/api/trainer/jobs/{j['job']}/adapter", content=data, headers=T)
    assert r.status_code == 200 and r.json()["adapter"] == aid
    assert (tmp / "adapters" / f"{aid}.npz").read_bytes() == data
    d = c.get("/api/me/doppel", headers=auth(x["token"])).json()
    assert d["state"] == "ready" and d["adapter"] == aid and d["package"] == "maia3-79m-lora8"
    assert c.post(f"/api/trainer/jobs/{j['job']}/adapter", content=data, headers=T).status_code == 410  # used up


def test_bad_adapters_refused(setup):
    c, fake, x, tmp = setup
    j = c.post("/api/trainer/claim", headers=T).json()
    _, other = adapter_bytes(tmp, package="maia3-23m-lora8")
    assert c.post(f"/api/trainer/jobs/{j['job']}/adapter", content=other, headers=T).status_code == 422
    assert c.post(f"/api/trainer/jobs/{j['job']}/adapter", content=b"junk", headers=T).status_code == 422
    assert c.post(f"/api/trainer/jobs/{j['job']}/adapter", content=b"x" * (2 << 20), headers=T).status_code == 413
    assert c.post("/api/trainer/jobs/../adapter", content=b"x", headers=T).status_code in (404, 405)
    assert c.get("/api/me/doppel", headers=auth(x["token"])).json()["state"] == "training"   # still waiting


def test_fail_and_timeout(setup):
    c, fake, x, tmp = setup
    j = c.post("/api/trainer/claim", headers=T).json()
    assert c.post(f"/api/trainer/jobs/{j['job']}/fail", json={"error": "cuda oom"}, headers=T).status_code == 200
    d = c.get("/api/me/doppel", headers=auth(x["token"])).json()
    assert d["state"] == "error" and "cuda" not in d["error"] and d["can_request"]
    assert c.post("/api/me/doppel", json={}, headers=auth(x["token"])).status_code == 202
    j = c.post("/api/trainer/claim", headers=T).json()
    with app.state.sessions() as sess:
        sess.execute(update(Doppel).values(claimed_at=time.time() - 3600))
        sess.commit()
    assert c.post("/api/trainer/claim", headers=T).json()["job"] is None
    assert c.get("/api/me/doppel", headers=auth(x["token"])).json()["state"] == "error"
    _, data = adapter_bytes(tmp)
    assert c.post(f"/api/trainer/jobs/{j['job']}/adapter", content=data, headers=T).status_code == 410


def test_deleted_meanwhile_and_daily_cap(setup, monkeypatch):
    c, fake, x, tmp = setup
    monkeypatch.setenv("DOPPEL_TRAIN_DAILY", "1")
    j = c.post("/api/trainer/claim", headers=T).json()
    r = c.put("/api/me/history-consent", json={"granted": False, "version": "history-v1"}, headers=auth(x["token"]))
    assert r.status_code == 200
    aid, data = adapter_bytes(tmp)
    assert c.post(f"/api/trainer/jobs/{j['job']}/adapter", content=data, headers=T).status_code == 410
    assert not (tmp / "adapters" / f"{aid}.npz").exists()
    with app.state.sessions() as sess:      # another queued doppel: the cap (1 claim per 24 h) holds it back
        sess.add(Doppel(user_id=x["user"]["id"], state="queued", speed="blitz", games=[], claimed_at=None))
        sess.add(Doppel(user_id="other", state="ready", speed="blitz", games=[], claimed_at=time.time() - 60))
        sess.commit()
    assert c.post("/api/trainer/claim", headers=T).json() == {"job": None, "reason": "daily cap"}
