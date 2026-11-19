"""Personal doppel requests (doppel.py) and the Lichess checks under them (history.py): flagged, closed and BOT
accounts refused; games against BOTs and the AI not counted; opp_bot backfilled; the doppel deleted with the
history, adapter file included."""

import time

from sqlalchemy import select, update
from test_history import T0, lgame, opt_in, run_import, signed_in
from test_lichess_login import auth, env  # noqa: F401 (env is a fixture)

from doppel_api import doppel
from doppel_api.app import app
from doppel_api.db import Doppel, ImportedGame

BOT = {"user": {"name": "Stock", "id": "stock", "title": "BOT"}, "rating": 2000}
AI = {"aiLevel": 3}


def ready(c, fake, games):
    fake.games["ann"] = games
    x = signed_in(c, fake)
    opt_in(c, x["token"])
    return x, run_import(c, x["token"])


def test_flagged_closed_and_bot_accounts_refused(env):
    c, fake = env
    for extra, words in (({"tosViolation": True}, "flagged"), ({"disabled": True}, "closed"),
                         ({"title": "BOT"}, "BOT accounts"), ("missing", "no such account")):
        fake.accounts["ann"] = extra
        x, s = ready(c, fake, [lgame("g1", T0)])
        assert s["state"] == "error" and words in s["error"] and s["games"] == 0, s


def test_counts_only_rated_games_against_people(env):
    c, fake = env
    gs = [lgame(f"h{i}", T0 + i) for i in range(5)]
    gs += [lgame("b1", T0 + 10, opp=BOT), lgame("a1", T0 + 11, opp=AI, rated=False),
           lgame("c1", T0 + 12, rated=False), lgame("r1", T0 + 13, speed="rapid", perf="rapid")]
    x, s = ready(c, fake, gs)
    assert s["games"] == 9
    d = c.get("/api/me/doppel", headers=auth(x["token"])).json()
    assert d["eligible"] == {"bullet": 0, "blitz": 5, "rapid": 1, "classical": 0} and d["best_speed"] == "blitz"
    assert d["state"] == "none" and not d["can_request"] and "needs 40" in d["reason"] and "5 at blitz" in d["reason"]
    assert c.post("/api/me/doppel", json={}, headers=auth(x["token"])).status_code == 403


def test_request_queues_most_recent_games_then_limits(env, monkeypatch):
    c, fake = env
    x, _ = ready(c, fake, [lgame(f"g{i:03d}", T0 + i * 1000) for i in range(230)])
    r = c.post("/api/me/doppel", json={}, headers=auth(x["token"]))
    assert r.status_code == 202, r.text
    d = r.json()
    assert d["state"] == "queued" and d["speed"] == "blitz" and d["games_used"] == 200
    with app.state.sessions() as sess:
        row = sess.get(Doppel, x["user"]["id"])
        assert row.games[0] == "g030" and row.games[-1] == "g229"   # the newest 200, oldest first
    r = c.post("/api/me/doppel", json={}, headers=auth(x["token"]))
    assert r.status_code == 403 and "already in training" in r.json()["detail"]
    with app.state.sessions() as sess:
        sess.execute(update(Doppel).values(state="ready", adapter_id="ab" * 8))
        sess.commit()
    r = c.post("/api/me/doppel", json={}, headers=auth(x["token"]))
    assert r.status_code == 403 and "per week" in r.json()["detail"]
    with app.state.sessions() as sess:
        sess.execute(update(Doppel).values(state="error"))
        sess.commit()
    assert c.post("/api/me/doppel", json={}, headers=auth(x["token"])).status_code == 202   # a failure can retry
    assert c.post("/api/me/doppel", json={"speed": "chess960"}, headers=auth(x["token"])).status_code == 422


def test_account_checked_again_at_request_time(env):
    c, fake = env
    x, _ = ready(c, fake, [lgame(f"g{i}", T0 + i) for i in range(40)])
    fake.accounts["ann"] = {"tosViolation": True}
    r = c.post("/api/me/doppel", json={}, headers=auth(x["token"]))
    assert r.status_code == 403 and "flagged" in r.json()["detail"]


def test_opp_bot_backfilled_by_a_full_refetch(env):
    c, fake = env
    x, _ = ready(c, fake, [lgame("g1", T0), lgame("g2", T0 + 1, opp=BOT)])
    with app.state.sessions() as sess:
        sess.execute(update(ImportedGame).values(opp_bot=None))   # as imported before opp_bot existed
        sess.commit()
    s = run_import(c, x["token"])
    assert "since" not in fake.export_calls[-1][1] and s["games"] == 2
    with app.state.sessions() as sess:
        got = dict(sess.execute(select(ImportedGame.lichess_game_id, ImportedGame.opp_bot)).all())
    assert got == {"g1": False, "g2": True}
    run_import(c, x["token"])
    assert "since" in fake.export_calls[-1][1]   # incremental again once backfilled


def test_doppel_and_adapter_deleted_with_history(env, tmp_path, monkeypatch):
    c, fake = env
    monkeypatch.setenv("DOPPEL_ADAPTERS", str(tmp_path))
    x, _ = ready(c, fake, [lgame(f"g{i}", T0 + i) for i in range(40)])
    assert c.post("/api/me/doppel", json={}, headers=auth(x["token"])).status_code == 202
    aid = "cd" * 8
    (tmp_path / f"{aid}.npz").write_bytes(b"x")
    with app.state.sessions() as sess:
        sess.execute(update(Doppel).values(state="ready", adapter_id=aid))
        sess.commit()
    r = c.put("/api/me/history-consent", json={"granted": False, "version": "history-v1"}, headers=auth(x["token"]))
    assert r.status_code == 200
    assert not (tmp_path / f"{aid}.npz").exists()
    assert c.get("/api/me/doppel", headers=auth(x["token"])).json()["state"] == "none"


def test_min_games_floor():
    import os
    os.environ["DOPPEL_MIN_GAMES"] = "3"
    try:
        assert doppel.min_games() == 15
    finally:
        del os.environ["DOPPEL_MIN_GAMES"]
