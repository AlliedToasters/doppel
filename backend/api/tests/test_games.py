"""API skeleton: server-authoritative games against a fake infer (first legal move, or a scripted one)."""

import time

import chess
import numpy as np
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
        return sorted(m.uci() for m in b.legal_moves)[0], 0.5


@pytest.fixture
def client(tmp_path):
    app.state.sessions = make_sessionmaker(f"sqlite:///{tmp_path}/t.db")
    app.state.infer = FakeInfer()
    with TestClient(app) as c:
        yield c


def test_create_white_and_move(client):
    g = client.post("/api/games", json={"color": "white", "pace": "fast"}).json()
    assert g["ply"] == 0 and g["turn"] == "white" and g["package_id"] == "maia3-79m-deadbeef"
    g2 = client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4", "ply": 0}).json()
    assert g2["ply"] == 2 and g2["moves"][0] == "e2e4" and g2["turn"] == "white"
    assert g2["clock"]["running"]  # clocks start once both sides moved


def test_black_gets_bot_first_move(client):
    g = client.post("/api/games", json={"color": "black", "pace": "fast"}).json()
    assert g["ply"] == 1 and g["turn"] == "black"


def test_illegal_stale_and_wrong_turn(client):
    g = client.post("/api/games", json={"color": "white", "pace": "fast"}).json()
    assert client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e5"}).status_code == 400
    client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4", "ply": 0})
    assert client.post(f"/api/games/{g['id']}/move", json={"uci": "d2d4", "ply": 0}).status_code == 409
    assert client.get("/api/games/nope").status_code == 404


def test_resign_and_resume(client):
    g = client.post("/api/games", json={"color": "white", "pace": "fast"}).json()
    client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4"})
    again = client.get(f"/api/games/{g['id']}").json()  # a fresh client resumes from server state
    assert again["moves"][0] == "e2e4"
    r = client.post(f"/api/games/{g['id']}/resign").json()
    assert (r["status"], r["result"], r["termination"]) == ("over", "0-1", "resign")
    assert client.post(f"/api/games/{g['id']}/move", json={"uci": "d2d4"}).status_code == 400


def test_leaving_before_both_moved_aborts(client):
    g = client.post("/api/games", json={"color": "white", "pace": "fast"}).json()
    r = client.post(f"/api/games/{g['id']}/resign").json()
    assert (r["status"], r["result"], r["termination"]) == ("over", "*", "abort")


def test_z_sampled_once_and_sent_every_move(tmp_path):
    app.state.sessions = make_sessionmaker(f"sqlite:///{tmp_path}/z.db")
    app.state.infer = fake = FakeInfer(z_prior={"type": "normal", "dim": 4, "scale": 1.0})
    with TestClient(app) as c:
        g = c.post("/api/games", json={"color": "white", "pace": "fast"}).json()
        for u in ("e2e4", "d2d4"):
            g = c.post(f"/api/games/{g['id']}/move", json={"uci": u}).json()
    zs = [z for _, z in fake.calls]
    assert len(zs) == 2 and len(zs[0]) == 4 and zs[0] == zs[1]


def test_websocket_move_then_bot_reply(client):
    g = client.post("/api/games", json={"color": "white", "pace": "fast"}).json()
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


def test_castling_either_notation_stored_canonical():
    for sent in ("e1g1", "e1h1"):
        g = _row()
        for u in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"):
            rules.play(g, u, 0.0)
        rules.play(g, sent, 0.0)
        assert g.moves[-1] == "e1g1"


def test_think_time_bounds():
    rng = np.random.default_rng(0)
    g = _row(base_ms=180000, inc_ms=2000, white_ms=180000, black_ms=180000)
    for u in ("e2e4", "e7e5"):
        rules.play(g, u, 0.0)
    assert all(400 <= rules.think_ms(g, 0.5, 0.0, rng) <= 1500 for _ in range(50))  # opening: quick
    g.moves = g.moves + ["g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"]  # bot (white) to move, own move 4
    ts = [rules.think_ms(g, 0.3, 0.0, rng) for _ in range(400)]
    assert 2000 < float(np.median(ts)) < 12000 and max(ts) <= 0.25 * 180000
    g.white_ms = 3000  # time trouble: never think itself into a flag
    assert max(rules.think_ms(g, 0.1, 0.0, rng) for _ in range(200)) <= 1500


def test_fast_pace_charges_think_time(client, monkeypatch):
    monkeypatch.setattr(rules, "think_ms", lambda *a: 4000)
    g = client.post("/api/games", json={"color": "white", "pace": "fast", "base_s": 60, "inc_s": 0}).json()
    t = time.time()
    g = client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4"}).json()
    g = client.post(f"/api/games/{g['id']}/move", json={"uci": "d2d4"}).json()  # clocks now running
    assert time.time() - t < 1.0  # no wall-clock wait
    assert g["clock"]["black_ms"] == 56000  # but the bot's clock paid 4 s


def test_realtime_pace_waits_and_switch_to_fast_wakes(client, monkeypatch):
    monkeypatch.setattr(rules, "think_ms", lambda *a: 600)
    g = client.post("/api/games", json={"color": "white", "pace": "realtime"}).json()
    t = time.time()
    g = client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4"}).json()
    assert g["ply"] == 2 and time.time() - t >= 0.55
    monkeypatch.setattr(rules, "think_ms", lambda *a: 30000)
    with client.websocket_connect(f"/api/games/{g['id']}/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "move", "uci": "d2d4", "ply": 2})
        assert ws.receive_json()["game"]["ply"] == 3
        t = time.time()
        ws.send_json({"type": "pace", "pace": "fast"})
        assert ws.receive_json()["game"]["pace"] == "fast"
        s = ws.receive_json()["game"]
        assert s["ply"] == 4 and time.time() - t < 2.0  # woke immediately instead of waiting 30 s


def test_plies_recorded_and_exported(client, monkeypatch):
    from doppel_api.export import run
    monkeypatch.setattr(rules, "think_ms", lambda *a: 2500)
    g = client.post("/api/games", json={"color": "white", "pace": "fast", "base_s": 60, "player": "dev1"}).json()
    with client.websocket_connect(f"/api/games/{g['id']}/ws") as ws:
        ws.receive_json()
        for ply, u, pre in ((0, "e2e4", False), (2, "d2d4", True)):
            ws.send_json({"type": "move", "uci": u, "ply": ply, "think_ms": 1234, "premove": pre})
            ws.receive_json(), ws.receive_json()
        ws.send_json({"type": "resign"})
        ws.receive_json()
    assert run(app.state.sessions)[0] == []  # pre-account game: never exported
    tok = client.post("/api/users", json={"install_id": "dev1"}).json()["token"]  # claims the game
    client.put("/api/me/consent", json={"training": True, "adult": True}, headers={"authorization": f"Bearer {tok}"})
    (r,), _ = run(app.state.sessions, salt="s")
    assert r["white"].startswith("doppel:") and "dev1" not in r["white"] and r["black"] == "maia3-79m-deadbeef"
    assert r["source"] == "doppel_native" and r["consent_version"] == "training-v2"
    assert r["actor"] == ["human", "bot", "human", "bot"]
    assert r["think_source"][1] == "heuristic_v1" and r["client_ms"][0] == 1234 and r["premove"] == [False, False, True, False]
    assert r["spent_ms"][:2] == [0, 0] and r["spent_ms"][3] == 2500  # clocks start after both sides moved
    assert r["clocks"][3] == 57.5 and len(r["clocks"]) == len(r["moves"]) == 4
