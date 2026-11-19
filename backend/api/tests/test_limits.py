"""Request limits for the open internet (doppel_api/limits.py)."""

import pytest
from fastapi.testclient import TestClient

from doppel_api import limits
from doppel_api.app import app
from doppel_api.db import make_sessionmaker
from test_games import FakeInfer


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_bucket_burst_then_refill():
    clk = Clock()
    lim = limits.Limiter({"r": (1.0, 3)}, clock=clk)
    assert [lim.wait("r", "a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert lim.wait("r", "a") == pytest.approx(1.0)
    assert lim.wait("r", "b") == 0.0          # keys are independent
    clk.t = 1.0
    assert lim.wait("r", "a") == 0.0
    assert lim.wait("r", "a") > 0


def test_disabled_never_refuses():
    lim = limits.Limiter({"r": (0.001, 1)}, enabled=False)
    assert all(lim.wait("r", "a") == 0.0 for _ in range(100))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DOPPEL_LIMITS", "on")
    app.state.sessions = make_sessionmaker(f"sqlite:///{tmp_path}/t.db")
    app.state.infer = FakeInfer()
    with TestClient(app) as c:
        yield c


def test_signup_per_ip(client):
    for _ in range(10):
        assert client.post("/api/users", json={}).status_code == 200
    r = client.post("/api/users", json={})
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    assert client.get("/api/health").status_code == 200   # other endpoints still answer


def test_ip_header_trusted_only_when_configured(client, monkeypatch):
    client.app.state.limits = limits.Limiter({**limits.RULES, "signup_ip": (1e-6, 1)})
    hdr = lambda ip: {"x-doppel-client-ip": ip}
    assert client.post("/api/users", json={}, headers=hdr("1.1.1.1")).status_code == 200
    assert client.post("/api/users", json={}, headers=hdr("2.2.2.2")).status_code == 429   # header ignored
    monkeypatch.setenv("DOPPEL_TRUST_IP_HEADER", "1")
    assert client.post("/api/users", json={}, headers=hdr("3.3.3.3")).status_code == 200
    assert client.post("/api/users", json={}, headers=hdr("3.3.3.3")).status_code == 429


def test_moves_per_account_rest_and_ws(client):
    client.app.state.limits = limits.Limiter({**limits.RULES, "move": (1e-6, 2)})
    g = client.post("/api/games", json={"color": "white", "pace": "fast"}).json()
    assert client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4", "ply": 0}).status_code == 200
    assert client.post(f"/api/games/{g['id']}/move", json={"uci": "d2d4", "ply": 2}).status_code == 200
    r = client.post(f"/api/games/{g['id']}/move", json={"uci": "g1f3", "ply": 4})
    assert r.status_code == 429
    with client.websocket_connect(f"/api/games/{g['id']}/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        ws.send_json({"type": "sync"})
        m = ws.receive_json()
        assert m["type"] == "error" and m["status"] == 429


def test_games_per_account(client):
    client.app.state.limits = limits.Limiter({**limits.RULES, "game": (1e-6, 2)})
    for _ in range(2):
        assert client.post("/api/games", json={"color": "white", "pace": "fast"}).status_code == 200
    assert client.post("/api/games", json={"color": "white", "pace": "fast"}).status_code == 429
