"""The training export against the other ways data reaches an account: the Lichess history import and the
anonymous-account merge at Lichess sign-in. Only 18+ training consent (training-v2) opens the export."""

from test_accounts import play_and_resign
from test_history import T0, lgame, opt_in, run_import, signed_in, stored
from test_lichess_login import auth, env  # noqa: F401 (env is a fixture)

from doppel_api.app import app
from doppel_api.export import run


def exported():
    rows, tomb = run(app.state.sessions)
    return [r["game_id"] for r in rows], tomb


def test_history_consent_alone_exports_nothing(env):
    c, fake = env
    fake.games["ann"] = [lgame("g1", T0), lgame("g2", T0 + 1000)]
    x = signed_in(c, fake)
    opt_in(c, x["token"])
    assert run_import(c, x["token"])["state"] == "done" and len(stored(x["user"]["id"])) == 2
    gid = play_and_resign(c, x["token"])
    assert exported() == ([], [])          # 18+ history consent is not training consent
    c.put("/api/me/consent", json={"training": True, "adult": True}, headers=auth(x["token"]))
    ids, _ = exported()
    assert ids == [gid]                    # played games only: imported Lichess games are not in the export yet
    assert not {"g1", "g2"} & set(ids)


def test_merge_follows_the_existing_accounts_consent(env):
    c, fake = env
    first = signed_in(c, fake)             # existing Lichess-linked account, no training consent
    anon = c.post("/api/users", json={}).json()
    c.put("/api/me/consent", json={"training": True, "adult": True}, headers=auth(anon["token"]))
    gid = play_and_resign(c, anon["token"])
    assert exported() == ([gid], [])       # exported while the anonymous account consented
    x = signed_in(c, fake, headers=auth(anon["token"]))
    assert x["user"]["id"] == first["user"]["id"] and x["user"]["consent"]["training"] is False
    assert x["user"]["games"] == 1         # the game moved over...
    assert exported() == ([], [gid])       # ...and is tombstoned, since the merged account never consented


def test_training_consent_is_18_plus(env):
    c, fake = env
    tok = c.post("/api/users", json={}).json()["token"]
    r = c.put("/api/me/consent", json={"training": True}, headers=auth(tok))
    assert r.status_code == 422 and "18" in r.json()["detail"]
    assert c.get("/api/me", headers=auth(tok)).json()["consent"]["training"] is False
    assert "18 or older" in c.get("/api/consent").json()["training"]


def test_v1_grants_without_the_age_flag_stop_exporting(env):
    from doppel_api.db import Consent
    c, fake = env
    x = c.post("/api/users", json={}).json()
    c.put("/api/me/consent", json={"training": True, "adult": True}, headers=auth(x["token"]))
    gid = play_and_resign(c, x["token"])
    assert exported() == ([gid], [])
    with app.state.sessions() as sess:     # as granted before the age check: training-v1, no 18+ flag
        row = sess.query(Consent).filter_by(user_id=x["user"]["id"], kind="training").one()
        row.version, row.adult = "training-v1", None
        sess.commit()
    assert c.get("/api/me", headers=auth(x["token"])).json()["consent"]["training"] is False
    assert exported() == ([], [gid])       # tombstoned until they grant training-v2 with the box ticked


def test_kid_mode_refused_and_stops_export(env):
    c, fake = env
    fake.kids["ann"] = True
    tok = signed_in(c, fake)["token"]
    r = c.put("/api/me/consent", json={"training": True, "adult": True}, headers=auth(tok))
    assert r.status_code == 403 and "kid" in r.json()["detail"]
    fake.kids["ann"] = False
    signed_in(c, fake)
    c.put("/api/me/consent", json={"training": True, "adult": True}, headers=auth(tok))
    gid = play_and_resign(c, tok)
    assert exported() == ([gid], [])
    fake.kids["ann"] = True                # turned kid mode on at Lichess, then signed in again
    signed_in(c, fake)
    assert exported() == ([], [gid])
