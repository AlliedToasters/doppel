"""API skeleton: server-authoritative games against a fake infer (first legal move, or a scripted one)."""

import time

import chess
import pytest
from fastapi.testclient import TestClient

from doppel_api import rules
from doppel_api.app import app
from doppel_api.db import Game, make_sessionmaker


class FakeInfer:
    def __init__(self, z_prior=None):
        self.z_prior, self.calls = z_prior, []

    async def bots(self):
        return {"maia3-79m": {"id": "maia3-79m", "package": "maia3-79m-deadbeef", "z_prior": self.z_prior}}

    async def move(self, g):
        self.calls.append((list(g.moves), g.z))
        b = chess.Board()
        for u in g.moves:
            b.push_uci(u)
        return sorted(m.uci() for m in b.legal_moves)[0]


@pytest.fixture
def client(tmp_path):
    app.state.sessions = make_sessionmaker(f"sqlite:///{tmp_path}/t.db")
    app.state.infer = FakeInfer()
    with TestClient(app) as c:
        yield c


def test_create_white_and_move(client):
    g = client.post("/api/games", json={"color": "white"}).json()
    assert g["ply"] == 0 and g["turn"] == "white" and g["package_id"] == "maia3-79m-deadbeef"
    g2 = client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4", "ply": 0}).json()
    assert g2["ply"] == 2 and g2["moves"][0] == "e2e4" and g2["turn"] == "white"
    assert g2["clock"]["running"]  # clocks start once both sides moved


def test_black_gets_bot_first_move(client):
    g = client.post("/api/games", json={"color": "black"}).json()
    assert g["ply"] == 1 and g["turn"] == "black"


def test_illegal_stale_and_wrong_turn(client):
    g = client.post("/api/games", json={"color": "white"}).json()
    assert client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e5"}).status_code == 400
    client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4", "ply": 0})
    assert client.post(f"/api/games/{g['id']}/move", json={"uci": "d2d4", "ply": 0}).status_code == 409
    assert client.get("/api/games/nope").status_code == 404


def test_resign_and_resume(client):
    g = client.post("/api/games", json={"color": "white"}).json()
    client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4"})
    again = client.get(f"/api/games/{g['id']}").json()  # a fresh client resumes from server state
    assert again["moves"][0] == "e2e4"
    r = client.post(f"/api/games/{g['id']}/resign").json()
    assert (r["status"], r["result"], r["termination"]) == ("over", "0-1", "resign")
    assert client.post(f"/api/games/{g['id']}/move", json={"uci": "d2d4"}).status_code == 400


def test_z_sampled_once_and_sent_every_move(tmp_path):
    app.state.sessions = make_sessionmaker(f"sqlite:///{tmp_path}/z.db")
    app.state.infer = fake = FakeInfer(z_prior={"type": "normal", "dim": 4, "scale": 1.0})
    with TestClient(app) as c:
        g = c.post("/api/games", json={"color": "white"}).json()
        for u in ("e2e4", "d2d4"):
            g = c.post(f"/api/games/{g['id']}/move", json={"uci": u}).json()
    zs = [z for _, z in fake.calls]
    assert len(zs) == 2 and len(zs[0]) == 4 and zs[0] == zs[1]


def test_websocket_move_then_bot_reply(client):
    g = client.post("/api/games", json={"color": "white"}).json()
    with client.websocket_connect(f"/api/games/{g['id']}/ws") as ws:
        assert ws.receive_json()["game"]["ply"] == 0
        ws.send_json({"type": "move", "uci": "e2e4", "ply": 0})
        assert ws.receive_json()["game"]["ply"] == 1  # own move echoed first
        assert ws.receive_json()["game"]["ply"] == 2  # then the bot's reply
        ws.send_json({"type": "move", "uci": "e2e5", "ply": 2})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "resign"})
        assert ws.receive_json()["game"]["termination"] == "resign"


def _row(**kw):
    base = dict(id="g", bot="b", package_id="p", bot_elo=1500, human_elo=1500, human_color="white",
                base_ms=60000, inc_ms=2000, white_ms=60000, black_ms=60000, turn_started=None, moves=[], z=None,
                status="active", result=None, termination=None)
    return Game(**(base | kw))


def test_clock_charge_increment_and_flag():
    g, t = _row(), 1000.0
    rules.play(g, "e2e4", t)
    rules.play(g, "e7e5", t + 5)  # clocks not running yet: no charge, no increment
    assert (g.white_ms, g.black_ms) == (60000, 60000) and g.turn_started == t + 5
    rules.play(g, "g1f3", t + 15)  # white spent 10 s, +2 s increment
    assert g.white_ms == 52000
    assert not rules.settle(g, t + 15 + 59)
    assert rules.settle(g, t + 15 + 61) and (g.result, g.termination) == ("1-0", "timeout")


def test_flag_against_lone_king_is_draw(monkeypatch):
    g = _row(moves=["e2e4", "e7e5"], turn_started=1000.0, white_ms=1000)  # white to move, 1 s left
    monkeypatch.setattr(rules, "board", lambda _g: chess.Board("8/8/8/8/8/5k2/8/4K3 w - - 0 1"))
    assert rules.settle(g, 1002.0) and (g.result, g.termination) == ("1/2-1/2", "timeout_vs_insufficient")
