"""Consented Lichess history import against the fake lichess.org: who may import (Lichess link, kid mode,
history-v1 with 18+), parsing the ndjson export, incremental `since`, the 429 backoff, one stream at a time,
deletion on withdraw and DELETE /api/me, and the anonymous-account merge."""

import asyncio
import time

import httpx
import pytest
from sqlalchemy import select
from test_lichess_login import FakeLichess, auth, env, login_code, sign_in  # noqa: F401 (env is a fixture)

from doppel_api import history
from doppel_api.app import app
from doppel_api.db import ImportedGame

T0 = 1_700_000_000_000   # ms


def lgame(gid, created, color="white", uid="ann", winner="white", **kw):
    me = {"user": {"name": "Ann", "id": uid}, "rating": 1500}
    opp = kw.pop("opp", {"user": {"name": "Bob", "id": "bob"}, "rating": 1600})
    g = {"id": gid, "rated": True, "variant": "standard", "speed": "blitz", "perf": "blitz", "createdAt": created,
         "lastMoveAt": created + 60_000, "status": "mate", "source": "pool",
         "players": {"white": me, "black": opp} if color == "white" else {"white": opp, "black": me},
         "moves": "e4 e5 Qh5 Nc6 Bc4 Nf6 Qxf7#", "clocks": [18003, 18003, 17800, 17700, 17500, 17400, 17300],
         "clock": {"initial": 180, "increment": 2, "totalTime": 260},
         "opening": {"eco": "C20", "name": "King's Pawn Game: Wayward Queen Attack", "ply": 3}}
    if winner:
        g["winner"] = winner
    g.update(kw)
    return g


def signed_in(c, fake, headers=None):
    return c.post("/api/auth/lichess/exchange", json={"code": login_code(sign_in(c, fake))},
                  headers=headers or {}).json()


def opt_in(c, tok):
    r = c.put("/api/me/history-consent", json={"granted": True, "version": "history-v1", "adult": True},
              headers=auth(tok))
    assert r.status_code == 200, r.text
    return r.json()


def run_import(c, tok, timeout=5.0):
    r = c.post("/api/me/history/import", headers=auth(tok))
    assert r.status_code == 202, r.text
    end = time.time() + timeout
    while time.time() < end:
        s = c.get("/api/me/history", headers=auth(tok)).json()
        if s["state"] in ("done", "error"):
            return s
        time.sleep(0.02)
    raise AssertionError(f"import did not finish: {s}")


def stored(user_id):
    with app.state.sessions() as sess:
        return sess.scalars(select(ImportedGame).where(ImportedGame.user_id == user_id)
                            .order_by(ImportedGame.played_at)).all()


def test_consent_text_served_and_versioned(env):
    c, fake = env
    t = c.get("/api/consent").json()
    assert t["version"] == "training-v2" and t["history_version"] == "history-v1"
    for words in ("public Lichess games", "personal doppel", "18 or older", "deletes the imported games"):
        assert words in t["history"]
    x = signed_in(c, fake)
    assert x["user"]["history_consent"]["granted"] is False
    r = c.put("/api/me/history-consent", json={"granted": True, "version": "old", "adult": True},
              headers=auth(x["token"]))
    assert r.status_code == 409
    me = opt_in(c, x["token"])
    assert me["history_consent"] == {"granted": True, "version": "history-v1", "adult": True,
                                     "current_version": "history-v1"}
    assert me["consent"]["training"] is False   # the training consent is separate


def test_refusals_no_link_no_consent_no_adult(env):
    c, fake = env
    anon = c.post("/api/users", json={}).json()["token"]
    assert c.put("/api/me/history-consent", json={"granted": True, "adult": True},
                 headers=auth(anon)).status_code == 403
    r = c.post("/api/me/history/import", headers=auth(anon))
    assert r.status_code == 403 and "Sign in with Lichess" in r.json()["detail"]
    assert c.post("/api/me/history/import").status_code == 401
    tok = signed_in(c, fake)["token"]
    r = c.post("/api/me/history/import", headers=auth(tok))
    assert r.status_code == 403 and "Turn on" in r.json()["detail"]
    assert c.put("/api/me/history-consent", json={"granted": True, "adult": False},
                 headers=auth(tok)).status_code == 422          # the 18+ box must be ticked
    s = c.get("/api/me/history", headers=auth(tok)).json()
    assert s["state"] == "none" and s["can_import"] is False and fake.export_calls == []


def test_kid_mode_refused(env):
    c, fake = env
    fake.kids["ann"] = True
    tok = signed_in(c, fake)["token"]
    r = c.put("/api/me/history-consent", json={"granted": True, "adult": True}, headers=auth(tok))
    assert r.status_code == 403 and "kid" in r.json()["detail"]
    r = c.post("/api/me/history/import", headers=auth(tok))
    assert r.status_code == 403 and "kid" in r.json()["detail"]
    assert fake.export_calls == []


def test_kid_flag_is_refreshed_at_sign_in_and_unknown_is_refused(env):
    c, fake = env
    tok = signed_in(c, fake)["token"]
    opt_in(c, tok)
    fake.kids["ann"] = True                       # turned kid mode on at Lichess, then signed in again
    signed_in(c, fake)
    r = c.post("/api/me/history/import", headers=auth(tok))
    assert r.status_code == 403 and "kid" in r.json()["detail"]
    fake.kids["ann"] = "error"                    # Lichess would not say: refuse until known
    signed_in(c, fake)
    r = c.post("/api/me/history/import", headers=auth(tok))
    assert r.status_code == 403 and "again" in r.json()["detail"]
    assert fake.export_calls == []


def test_import_parses_ndjson(env):
    c, fake = env
    fake.games["ann"] = [
        lgame("g1", T0, color="white", winner="white"),
        lgame("g2", T0 + 1000, color="black", winner=None, status="draw", rated=False, speed="rapid", perf="rapid"),
        lgame("g3", T0 + 2000, color="black", opp={"aiLevel": 3}, winner="white", clocks=None, clock=None,
              opening=None, status="resign"),
        lgame("v1", T0 + 3000, variant="chess960", perf="chess960"),       # variant: skipped
        lgame("p1", T0 + 4000, variant="fromPosition", initialFen="8/8/8/8/8/8/8/K6k w - - 0 1"),
        lgame("a1", T0 + 5000, status="aborted", winner=None, moves=""),   # aborted: skipped
    ]
    x = signed_in(c, fake)
    opt_in(c, x["token"])
    s = run_import(c, x["token"])
    assert s["state"] == "done" and s["games"] == 3 and s["added"] == 3 and s["error"] is None
    path, q, h = fake.export_calls[0]
    assert path == "/api/games/user/ann" and h["accept"] == "application/x-ndjson"
    assert q["clocks"] == "true" and q["opening"] == "true" and q["moves"] == "true" and q["max"] == "5000"
    assert q["perfType"] == "ultraBullet,bullet,blitz,rapid,classical,correspondence" and "since" not in q
    assert "authorization" not in h and h["user-agent"].startswith("doppel-api")
    g1, g2, g3 = stored(x["user"]["id"])
    assert (g1.lichess_game_id, g1.color, g1.result, g1.rating, g1.opp_rating, g1.rated) == ("g1", "white", "1-0", 1500, 1600, True)
    assert g1.moves == ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"] and g1.clocks[0] == 18003
    assert (g1.base_s, g1.inc_s, g1.opening_eco, g1.speed, g1.status) == (180, 2, "C20", "blitz", "mate")
    assert g1.played_at == T0 / 1000
    assert (g2.color, g2.result, g2.rated, g2.speed, g2.status) == ("black", "1/2-1/2", False, "rapid", "draw")
    assert (g3.opp_rating, g3.clocks, g3.base_s, g3.opening_name, g3.result) == (None, None, None, None, "1-0")


def test_incremental_since_and_cap(env, monkeypatch):
    c, fake = env
    fake.games["ann"] = [lgame("g1", T0), lgame("g2", T0 + 1000)]
    x = signed_in(c, fake)
    opt_in(c, x["token"])
    assert run_import(c, x["token"])["games"] == 2
    fake.games["ann"].append(lgame("g3", T0 + 5000))
    s = run_import(c, x["token"])
    assert fake.export_calls[1][1]["since"] == str(T0 + 1000)   # newest stored game; repeats are skipped
    assert s["games"] == 3 and s["added"] == 1
    s = run_import(c, x["token"])
    assert fake.export_calls[2][1]["since"] == str(T0 + 5000) and s["added"] == 0 and s["games"] == 3
    monkeypatch.setenv("DOPPEL_HISTORY_MAX_GAMES", "2")
    run_import(c, x["token"])
    assert fake.export_calls[3][1]["max"] == "2"


def test_429_backs_off_at_least_a_minute(env, monkeypatch):
    c, fake = env
    slept = []

    async def fake_sleep(s):
        slept.append(s)
    monkeypatch.setattr(history, "_sleep", fake_sleep)
    fake.games["ann"] = [lgame("g1", T0)]
    fake.statuses = [429]
    x = signed_in(c, fake)
    opt_in(c, x["token"])
    s = run_import(c, x["token"])
    assert s["state"] == "done" and s["games"] == 1
    assert len(fake.export_calls) == 2 and len(slept) == 1 and slept[0] >= 59
    fake.statuses = [429, 429, 429]                   # gives up after three tries, with a plain message
    s = run_import(c, x["token"])
    # four waits: the backoff left by the first run (time is frozen here), then one after each 429 but the last
    assert s["state"] == "error" and "rate limit" in s["error"] and len(slept) == 4


def test_one_export_stream_at_a_time(env):
    c, fake = env
    fake.users["other"] = ("bob", "Bob")
    fake.games = {"ann": [lgame("a1", T0)], "bob": [lgame("b1", T0, uid="bob")]}
    active, peak = [0], [0]

    async def slow(req):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        await asyncio.sleep(0.1)
        active[0] -= 1
        return fake.handler(req)
    app.state.lichess = httpx.AsyncClient(base_url="https://lichess.test", transport=httpx.MockTransport(slow))
    a = signed_in(c, fake)["token"]
    r = sign_in(c, fake, code="other")
    b = c.post("/api/auth/lichess/exchange", json={"code": login_code(r)}).json()["token"]
    for t in (a, b):
        opt_in(c, t)
        assert c.post("/api/me/history/import", headers=auth(t)).status_code == 202
    assert c.get("/api/me/history", headers=auth(b)).json()["state"] in ("queued", "running")
    sa, sb = run_import(c, a), run_import(c, b)
    assert sa["games"] == 1 and sb["games"] == 1 and peak[0] == 1


def test_withdraw_deletes_imported_games(env):
    c, fake = env
    fake.games["ann"] = [lgame("g1", T0), lgame("g2", T0 + 1)]
    x = signed_in(c, fake)
    opt_in(c, x["token"])
    run_import(c, x["token"])
    me = c.put("/api/me/history-consent", json={"granted": False}, headers=auth(x["token"])).json()
    assert me["history_consent"]["granted"] is False and stored(x["user"]["id"]) == []
    s = c.get("/api/me/history", headers=auth(x["token"])).json()
    assert s["state"] == "none" and s["games"] == 0
    assert c.post("/api/me/history/import", headers=auth(x["token"])).status_code == 403
    opt_in(c, x["token"])                              # opting in again starts from scratch
    run_import(c, x["token"])
    assert "since" not in fake.export_calls[-1][1] and len(stored(x["user"]["id"])) == 2


def test_delete_account_deletes_imported_games(env):
    c, fake = env
    fake.games["ann"] = [lgame("g1", T0)]
    x = signed_in(c, fake)
    opt_in(c, x["token"])
    run_import(c, x["token"])
    r = c.delete("/api/me", headers=auth(x["token"])).json()
    assert r["deleted_imported_games"] == 1 and stored(x["user"]["id"]) == []


def test_merge_keeps_lichess_history_and_drops_anonymous_rows(env):
    c, fake = env
    fake.games["ann"] = [lgame("g1", T0)]
    first = signed_in(c, fake)
    opt_in(c, first["token"])
    run_import(c, first["token"])
    anon = c.post("/api/users", json={}).json()
    with app.state.sessions() as sess:    # cannot happen through the api; the merge must still not move it
        sess.add(ImportedGame(user_id=anon["user"]["id"], lichess_game_id="zz", played_at=1.0, speed="blitz",
                              perf="blitz", rated=True, color="white", status="mate", moves=["e4"]))
        sess.commit()
    x = signed_in(c, fake, headers=auth(anon["token"]))
    assert x["user"]["id"] == first["user"]["id"] and x["user"]["history_consent"]["granted"] is True
    assert [g.lichess_game_id for g in stored(first["user"]["id"])] == ["g1"]
    assert stored(anon["user"]["id"]) == []
    assert c.get("/api/me/history", headers=auth(x["token"])).json()["games"] == 1


def test_parse_skips_games_not_by_the_user():
    assert history.parse(lgame("g1", T0, uid="someone-else"), "ann") is None
    created, row = history.parse(lgame("g1", T0), "ann")
    assert created == T0 and row["lichess_game_id"] == "g1"


@pytest.mark.parametrize("state", ["queued", "running"])
def test_restart_marks_interrupted_runs(env, state):
    c, fake = env
    x = signed_in(c, fake)
    from doppel_api.db import HistoryImport
    with app.state.sessions() as sess:
        sess.add(HistoryImport(user_id=x["user"]["id"], state=state))
        sess.commit()
    app.state.history.recover()
    s = c.get("/api/me/history", headers=auth(x["token"])).json()
    assert s["state"] == "error" and "restart" in s["error"]
