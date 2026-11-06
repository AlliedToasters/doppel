import asyncio

import numpy as np

from doppel_lichess_bot.runner import Config, Game, decide


def ch(**kw):
    base = {"id": "c1", "challenger": {"id": "alice", "rating": 1400}, "variant": {"key": "standard"},
            "timeControl": {"type": "clock", "limit": 300, "increment": 3}, "rated": False}
    base.update(kw)
    return base


def test_decide():
    cfg = Config(allow={"alice"})
    assert decide(ch(), cfg, 0) is None
    assert decide(ch(challenger={"id": "bob"}), cfg, 0) == "later"
    assert decide(ch(challenger={"id": "Alice"}), cfg, 0) is None  # ids compare lowercase
    assert decide(ch(challenger={"id": "alice", "title": "BOT"}), cfg, 0) == "noBot"
    assert decide(ch(variant={"key": "chess960"}), cfg, 0) == "standard"
    assert decide(ch(timeControl={"type": "correspondence", "daysPerTurn": 3}), cfg, 0) == "timeControl"
    assert decide(ch(timeControl={"type": "clock", "limit": 30, "increment": 0}), cfg, 0) == "tooFast"
    assert decide(ch(rated=True), cfg, 0) == "casual"
    assert decide(ch(rated=True), Config(allow={"alice"}, rated=True), 0) is None
    assert decide(ch(), cfg, 2) == "later"
    assert decide(ch(challenger={"id": "anyone"}), Config(), 0) is None


class FakeResp:
    status_code, text = 200, ""


class FakeLichess:
    def __init__(self):
        self.posts = []

    async def post(self, path, data=None):
        self.posts.append((path, data))
        return FakeResp()


class FakeInfer:
    def __init__(self):
        self.calls = []

    async def move(self, bot, elo, opp_elo, moves):
        self.calls.append((elo, list(moves)))
        return ("e7e5" if moves == ["e2e4"] else "e2e4"), 0.6


async def nosleep(_):
    pass


def full(moves="", status="started", white="alice", black="doppelbot", rating=1234):
    return {"type": "gameFull", "white": {"id": white, "rating": rating}, "black": {"id": black, "rating": 1900},
            "clock": {"initial": 300000, "increment": 2000},
            "state": {"type": "gameState", "moves": moves, "wtime": 300000, "btime": 300000, "status": status}}


def test_game_moves_on_our_turn_only_once_per_ply():
    async def go():
        lc, inf = FakeLichess(), FakeInfer()
        g = Game(lc, inf, Config(), "doppelbot", "g1", rng=np.random.default_rng(0), sleep=nosleep)
        assert not await g.on_event(full())  # white to move: not ours
        assert g.color == "black" and g.elo == 1234
        st = {"type": "gameState", "moves": "e2e4", "wtime": 299000, "btime": 300000, "status": "started"}
        await g.on_event(st)
        await g.on_event(dict(st))  # duplicate state (e.g. draw-offer flag change): no second move
        await g.pending
        assert inf.calls == [(1234, ["e2e4"])]
        moves = [p for p, _ in lc.posts if "/move/" in p]
        assert moves == ["/api/bot/game/g1/move/e7e5"]
        assert any("/chat" in p for p, _ in lc.posts)  # greets once
        assert await g.on_event({"type": "gameState", "moves": "e2e4 e7e5", "status": "resign", "winner": "black"})
    asyncio.run(go())


def test_game_as_white_moves_first_and_clamps_rating():
    async def go():
        lc, inf = FakeLichess(), FakeInfer()
        g = Game(lc, inf, Config(), "doppelbot", "g2", sleep=nosleep)
        await g.on_event(full(white="doppelbot", black="newbie", rating=None) | {
            "black": {"id": "newbie", "rating": 400}})
        await g.pending
        assert g.color == "white" and g.elo == 600
        assert [p for p, _ in lc.posts if "/move/" in p] == ["/api/bot/game/g2/move/e2e4"]
    asyncio.run(go())
