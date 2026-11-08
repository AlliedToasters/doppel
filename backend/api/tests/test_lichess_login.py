"""Sign in with Lichess against a fake lichess.org: PKCE start, callback, token revoke, exchange and merging."""

import base64
import hashlib
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient
from test_games import FakeInfer

from doppel_api.app import app
from doppel_api.db import make_sessionmaker


class FakeLichess:
    def __init__(self):
        self.users = {"good": ("ann", "Ann")}   # code -> (id, username)
        self.verifiers, self.revoked, self.live = {}, [], set()

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/api/token":
            f = parse_qs(req.content.decode())
            code = f["code"][0]
            if code not in self.users:
                return httpx.Response(400, json={"error": "invalid_grant"})
            ch = base64.urlsafe_b64encode(hashlib.sha256(f["code_verifier"][0].encode()).digest()).rstrip(b"=").decode()
            assert ch == self.verifiers["challenge"]
            tok = f"lio_{code}"
            self.live.add(tok)
            return httpx.Response(200, json={"access_token": tok, "token_type": "Bearer"})
        tok = req.headers.get("authorization", "")[7:]
        if req.url.path == "/api/account" and tok in self.live:
            uid, name = self.users[tok[4:]]
            return httpx.Response(200, json={"id": uid, "username": name})
        if req.method == "DELETE" and req.url.path == "/api/token":
            self.live.discard(tok)
            self.revoked.append(tok)
            return httpx.Response(204)
        return httpx.Response(404)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DOPPEL_PUBLIC_URL", "https://doppel.test")
    fake = FakeLichess()
    app.state.sessions = make_sessionmaker(f"sqlite:///{tmp_path}/a.db")
    app.state.infer = FakeInfer()
    app.state.lichess = httpx.AsyncClient(base_url="https://lichess.test", transport=httpx.MockTransport(fake.handler))
    with TestClient(app, follow_redirects=False) as c:
        yield c, fake


def auth(tok):
    return {"authorization": f"Bearer {tok}"}


def sign_in(c, fake, code="good", client="web"):
    r = c.get("/api/auth/lichess/start", params={"client": client})
    assert r.status_code == 302
    q = parse_qs(urlparse(r.headers["location"]).query, keep_blank_values=True)
    assert q["scope"] == [""] and q["code_challenge_method"] == ["S256"]
    assert q["redirect_uri"] == ["https://doppel.test/api/auth/lichess/callback"]
    fake.verifiers["challenge"] = q["code_challenge"][0]
    return c.get("/api/auth/lichess/callback", params={"state": q["state"][0], "code": code})


def login_code(r):
    return parse_qs(urlparse(r.headers["location"]).query)["login_code"][0]


def test_new_lichess_account_and_token_revoked(env):
    c, fake = env
    r = sign_in(c, fake)
    assert r.headers["location"].startswith("https://doppel.test/account/?login_code=")
    assert fake.revoked == ["lio_good"] and not fake.live
    x = c.post("/api/auth/lichess/exchange", json={"code": login_code(r)}).json()
    assert x["user"]["lichess"] == {"id": "ann", "username": "Ann"} and x["user"]["display_name"] == "Ann"
    assert c.get("/api/me", headers=auth(x["token"])).json()["id"] == x["user"]["id"]


def test_code_is_single_use_and_state_checked(env):
    c, fake = env
    code = login_code(sign_in(c, fake))
    assert c.post("/api/auth/lichess/exchange", json={"code": code}).status_code == 200
    assert c.post("/api/auth/lichess/exchange", json={"code": code}).status_code == 400
    assert c.get("/api/auth/lichess/callback", params={"state": "forged", "code": "good"}).status_code == 400
    assert c.get("/api/auth/lichess/start", params={"client": "https://evil.test"}).status_code == 422


def test_app_client_and_errors_redirect_back(env):
    c, fake = env
    r = sign_in(c, fake, client="app")
    assert r.headers["location"].startswith("app.doppelchess.doppel://doppel-login?login_code=")
    r = sign_in(c, fake, code="bad")
    assert r.headers["location"] == "https://doppel.test/account/?login_error=lichess"


def test_anonymous_account_gets_linked_and_keeps_games(env):
    c, fake = env
    anon = c.post("/api/users", json={}).json()
    c.post("/api/games", json={"color": "white", "pace": "fast"}, headers=auth(anon["token"]))
    x = c.post("/api/auth/lichess/exchange", json={"code": login_code(sign_in(c, fake))},
               headers=auth(anon["token"])).json()
    assert x["user"]["id"] == anon["user"]["id"] and x["user"]["games"] == 1 and x["merged_games"] == 0
    assert c.get("/api/me", headers=auth(anon["token"])).status_code == 200   # old token still works


def test_second_device_merges_into_known_account(env):
    c, fake = env
    first = c.post("/api/auth/lichess/exchange", json={"code": login_code(sign_in(c, fake))}).json()
    anon = c.post("/api/users", json={}).json()
    c.post("/api/games", json={"color": "white", "pace": "fast"}, headers=auth(anon["token"]))
    x = c.post("/api/auth/lichess/exchange", json={"code": login_code(sign_in(c, fake))},
               headers=auth(anon["token"])).json()
    assert x["user"]["id"] == first["user"]["id"] and x["merged_games"] == 1 and x["user"]["games"] == 1
    assert c.get("/api/me", headers=auth(anon["token"])).status_code == 401   # anonymous account is gone
    assert c.get("/api/me", headers=auth(first["token"])).status_code == 200  # both devices signed in


def test_other_lichess_account_refused(env):
    c, fake = env
    fake.users["other"] = ("bob", "Bob")
    a = c.post("/api/auth/lichess/exchange", json={"code": login_code(sign_in(c, fake))}).json()
    r = c.post("/api/auth/lichess/exchange", json={"code": login_code(sign_in(c, fake, code="other"))},
               headers=auth(a["token"]))
    assert r.status_code == 409


def test_delete_account_removes_device_tokens(env):
    c, fake = env
    x = c.post("/api/auth/lichess/exchange", json={"code": login_code(sign_in(c, fake))}).json()
    assert c.delete("/api/me", headers=auth(x["token"])).status_code == 200
    assert c.get("/api/me", headers=auth(x["token"])).status_code == 401
    y = c.post("/api/auth/lichess/exchange", json={"code": login_code(sign_in(c, fake))}).json()
    assert y["user"]["id"] != x["user"]["id"] and y["user"]["games"] == 0
