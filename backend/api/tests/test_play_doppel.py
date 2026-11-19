"""Playing your own doppel (app.create_game with bot "doppel", doppel.playable): listed first in /api/bots only
for its owner, games go to infer with the adapter, the doppel plays at the user's median rating, and those games
are never rated."""

import httpx
import pytest
from sqlalchemy import update
from test_doppel_api import ready
from test_games import FakeInfer
from test_history import T0, lgame
from test_lichess_login import auth, env  # noqa: F401 (env is a fixture)

from doppel_api import ratings
from doppel_api.app import app
from doppel_api.db import Doppel, Game

AID = "ab12" * 4


class DoppelInfer(FakeInfer):
    def __init__(self):
        super().__init__()
        self.adapters = []
        bots = {"bots": [{"id": "maia3-79m", "name": "Maia-3 79M", "blurb": "", "clock": False, "z_prior": None,
                          "package": "maia3-79m-deadbeef"}]}
        self.http = httpx.AsyncClient(base_url="http://infer.test",
                                      transport=httpx.MockTransport(lambda r: httpx.Response(200, json=bots)))

    async def doppel_bases(self):
        return {"maia3-79m-lora8": "maia3-79m-lora8-cafef00d"}

    async def move(self, g):
        self.adapters.append(g.adapter_id)
        return await super().move(g)


@pytest.fixture
def mine(env):
    c, fake = env
    app.state.infer = DoppelInfer()
    games = [lgame(f"g{i:02d}", T0 + i * 1000) for i in range(41)]
    for i, g in enumerate(games):                       # ratings 1500..1540: median 1520
        side = "white" if g["players"]["white"]["user"]["id"] == "ann" else "black"
        g["players"][side]["rating"] = 1500 + i
    x, _ = ready(c, fake, games)
    assert c.post("/api/me/doppel", json={}, headers=auth(x["token"])).status_code == 202
    with app.state.sessions() as sess:
        sess.execute(update(Doppel).values(state="ready", adapter_id=AID, package="maia3-79m-lora8"))
        sess.commit()
    return c, fake, x


def test_listed_only_for_its_owner(mine):
    c, fake, x = mine
    b = c.get("/api/bots", headers=auth(x["token"])).json()["bots"]
    assert [i["id"] for i in b] == ["doppel", "maia3-79m"] and b[0]["name"] == "Your doppel"
    assert "41 of your blitz games" in b[0]["blurb"]
    assert [i["id"] for i in c.get("/api/bots").json()["bots"]] == ["maia3-79m"]          # anonymous
    other = c.post("/api/users", json={}).json()["token"]
    assert [i["id"] for i in c.get("/api/bots", headers=auth(other)).json()["bots"]] == ["maia3-79m"]


def test_play_your_doppel(mine):
    c, fake, x = mine
    r = c.post("/api/games", json={"bot": "doppel", "color": "black", "pace": "fast", "bot_elo": 2500},
               headers=auth(x["token"]))
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["ply"] == 1 and app.state.infer.adapters == [AID]
    with app.state.sessions() as sess:
        row = sess.get(Game, g["id"])
        assert row.bot == "maia3-79m-lora8" and row.package_id == f"maia3-79m-lora8-cafef00d+{AID}"
        assert row.bot_elo == 1520                                   # the user's median, not the request's 2500
        row.status, row.result, row.moves = "over", "1-0", ["e2e4"] * 20
        assert not ratings.rated(row)                                # games against yourself are not rated
    c.post(f"/api/games/{g['id']}/move", json={"uci": "e7e5", "ply": 1}, headers=auth(x["token"]))
    assert app.state.infer.adapters == [AID, AID]


def test_no_doppel_no_game(env):
    c, fake = env
    app.state.infer = DoppelInfer()
    tok = c.post("/api/users", json={}).json()["token"]
    r = c.post("/api/games", json={"bot": "doppel", "color": "white"}, headers=auth(tok))
    assert r.status_code == 403 and "no doppel" in r.json()["detail"]
