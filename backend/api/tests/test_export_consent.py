"""The training export against the other ways data reaches an account: the Lichess history import and the
anonymous-account merge at Lichess sign-in. Only training consent (training-v1) opens the export."""

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
    c.put("/api/me/consent", json={"training": True}, headers=auth(x["token"]))
    ids, _ = exported()
    assert ids == [gid]                    # played games only: imported Lichess games are not in the export yet
    assert not {"g1", "g2"} & set(ids)


def test_merge_follows_the_existing_accounts_consent(env):
    c, fake = env
    first = signed_in(c, fake)             # existing Lichess-linked account, no training consent
    anon = c.post("/api/users", json={}).json()
    c.put("/api/me/consent", json={"training": True}, headers=auth(anon["token"]))
    gid = play_and_resign(c, anon["token"])
    assert exported() == ([gid], [])       # exported while the anonymous account consented
    x = signed_in(c, fake, headers=auth(anon["token"]))
    assert x["user"]["id"] == first["user"]["id"] and x["user"]["consent"]["training"] is False
    assert x["user"]["games"] == 1         # the game moved over...
    assert exported() == ([], [gid])       # ...and is tombstoned, since the merged account never consented
