"""Accounts, ownership, consent, consented export with tombstones, ratings."""

import pytest
from fastapi.testclient import TestClient
from test_games import FakeInfer

from doppel_api import ratings
from doppel_api.app import app
from doppel_api.db import make_sessionmaker
from doppel_api.export import run


@pytest.fixture
def client(tmp_path):
    app.state.sessions = make_sessionmaker(f"sqlite:///{tmp_path}/a.db")
    app.state.infer = FakeInfer()
    with TestClient(app) as c:
        yield c


def auth(tok):
    return {"authorization": f"Bearer {tok}"}


def new_user(c, **kw):
    r = c.post("/api/users", json=kw).json()
    return r["token"], r


def play_and_resign(c, tok):
    g = c.post("/api/games", json={"color": "white", "pace": "fast"}, headers=auth(tok)).json()
    for u in ("e2e4", "d2d4"):
        c.post(f"/api/games/{g['id']}/move", json={"uci": u}, headers=auth(tok))
    c.post(f"/api/games/{g['id']}/resign", headers=auth(tok))
    return g["id"]


def test_user_token_and_me(client):
    tok, r = new_user(client, display_name="ann")
    assert r["user"]["display_name"] == "ann" and r["user"]["consent"]["training"] is False
    assert client.get("/api/me", headers=auth(tok)).json()["id"] == r["user"]["id"]
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/me", headers=auth("bogus")).status_code == 401


def test_install_id_claims_legacy_games_once(client):
    client.post("/api/games", json={"color": "white", "pace": "fast", "player": "dev9"})
    tok, r = new_user(client, install_id="dev9")
    assert r["claimed_games"] == 1 and r["user"]["games"] == 1
    assert client.post("/api/users", json={"install_id": "dev9"}).status_code == 409


def test_games_are_private_to_their_account(client):
    a, _ = new_user(client)
    b, _ = new_user(client)
    g = client.post("/api/games", json={"color": "white", "pace": "fast"}, headers=auth(a)).json()
    assert client.get(f"/api/games/{g['id']}", headers=auth(a)).status_code == 200
    for h in (auth(b), {}):
        assert client.get(f"/api/games/{g['id']}", headers=h).status_code == 403
        assert client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4"}, headers=h).status_code == 403
        assert client.post(f"/api/games/{g['id']}/resign", headers=h).status_code == 403
    with pytest.raises(Exception):
        with client.websocket_connect(f"/api/games/{g['id']}/ws?token={b}") as ws:
            ws.receive_json()
    with client.websocket_connect(f"/api/games/{g['id']}/ws?token={a}") as ws:
        assert ws.receive_json()["game"]["id"] == g["id"]
    mine = client.get("/api/me/games", headers=auth(a)).json()["games"]
    assert [x["id"] for x in mine] == [g["id"]]


def test_require_auth(client, monkeypatch):
    monkeypatch.setenv("DOPPEL_REQUIRE_AUTH", "1")
    assert client.post("/api/games", json={"pace": "fast"}).status_code == 401


def test_consent_versioned_and_withdrawable(client):
    tok, _ = new_user(client)
    assert client.get("/api/consent").json()["version"] == "training-v2"
    assert client.put("/api/me/consent", json={"training": True, "version": "old"}, headers=auth(tok)).status_code == 409
    me = client.put("/api/me/consent", json={"training": True, "adult": True}, headers=auth(tok)).json()
    assert me["consent"] == {"training": True, "version": "training-v2", "current_version": "training-v2"}
    me = client.put("/api/me/consent", json={"training": False}, headers=auth(tok)).json()
    assert me["consent"]["training"] is False


def test_export_consented_only_salted_and_tombstones(client):
    yes, _ = new_user(client)
    no, _ = new_user(client)
    client.put("/api/me/consent", json={"training": True, "adult": True}, headers=auth(yes))
    gy, gn = play_and_resign(client, yes), play_and_resign(client, no)
    r1, t1 = run(app.state.sessions)
    assert [r["game_id"] for r in r1] == [gy] and t1 == []
    r2, _ = run(app.state.sessions)
    assert r1[0]["white"] != r2[0]["white"]  # fresh salt per export: not linkable across exports
    client.put("/api/me/consent", json={"training": False}, headers=auth(yes))
    r3, t3 = run(app.state.sessions)
    assert r3 == [] and t3 == [gy]
    assert run(app.state.sessions)[1] == []  # a tombstone is sent once
    client.put("/api/me/consent", json={"training": True, "adult": True}, headers=auth(yes))
    assert [r["game_id"] for r in run(app.state.sessions)[0]] == [gy]
    assert client.delete("/api/me", headers=auth(yes)).json()["deleted_games"] == 1
    assert run(app.state.sessions) == ([], [gy])
    assert client.get("/api/me", headers=auth(yes)).status_code == 401
    assert gn  # the non-consenting user's game was never exported


def test_rating_refits_after_rated_game(client):
    tok, _ = new_user(client)
    for _ in range(3):
        play_and_resign(client, tok)
    ratings.refit(app.state.sessions)  # the background refit is fire-and-forget; run it deterministically
    r = client.get("/api/me", headers=auth(tok)).json()["rating"]
    assert r["games"] == 3 and r["rating"] < 1500 and r["rd"] < 350


def test_fit_recovers_order():
    pri = {"u": (1500.0, 350.0), "b1": (1200.0, 100.0), "b2": (1800.0, 100.0)}
    res = [("u", "b1", 1.0)] * 8 + [("u", "b2", 0.0)] * 8 + [("u", "b2", 0.5)] * 2
    out = ratings.fit(res, pri)
    assert 1200 < out["u"][0] < 1800 and out["b1"][0] < 1200 and out["b2"][0] > 1800


def test_tokens_never_reach_logs():
    import logging

    from doppel_api.accounts import RedactTokens
    rec = logging.LogRecord("uvicorn.error", logging.INFO, "", 0, '%s - "WebSocket %s" [accepted]',
                            ("1.2.3.4:0", "/api/games/abc/ws?token=s3cr3t-Tok_en&x=1"), None)
    RedactTokens().filter(rec)
    assert "s3cr3t" not in rec.getMessage() and "token=REDACTED&x=1" in rec.getMessage()


def test_no_stateless_move_endpoint(client):
    assert client.post("/api/move", json={"bot": "x", "moves": []}).status_code in (404, 405)


def test_redacted_access_lines_still_format():
    """uvicorn's access formatter unpacks record.args; redaction must keep that shape (the live Lichess callback
    line raised TypeError and was dropped)."""
    import logging

    from uvicorn.logging import AccessFormatter

    from doppel_api.accounts import RedactTokens
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                            ("1.2.3.4:0", "GET", "/api/auth/lichess/callback?state=st4te&code=c0de", "1.1", 302),
                            None)
    RedactTokens().filter(rec)
    line = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s').format(rec)
    assert "st4te" not in line and "c0de" not in line and "code=REDACTED" in line and "302" in line
