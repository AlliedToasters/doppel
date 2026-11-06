"""Doppel on Lichess: one BOT account plays every incoming challenge through backend/infer.

The doppel plays at the challenger's own rating for that speed (clamped to what the model covers), with
human-like think times. Lichess rules this runner keeps (ARCHITECTURE.md, Lichess hard boundary):
  - BOT account only (Bot API); engine play is allowed there and nowhere else.
  - Casual only unless --rated; no bot-vs-bot games; no sandbagging, no aborting on purpose.
  - --allow restricts who may challenge (testing, and until the AGPL source offer is public).

    .venv/bin/python -m doppel_lichess_bot.runner --allow <your lichess name>
Token: ~/.config/doppel/lichess_bot.token (bot:play), never logged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import httpx
import numpy as np

from doppel_model import heuristic_think_ms

from .ledger import Gate, Ledger

log = logging.getLogger("doppel.lichess_bot")
ELO_MIN, ELO_MAX = 600, 2600
MIN_BASE_S, MAX_BASE_S = 60, 3600
SOURCE_NOTE = "Open source (AGPL-3.0): https://github.com/AlliedToasters/doppel"  # AGPL section 13 offer


@dataclass
class Config:
    bot: str = "maia3-79m"
    allow: set[str] = field(default_factory=set)  # lowercase Lichess ids; empty = anyone
    rated: bool = False
    max_games: int = 2


def clamp_elo(r) -> int:
    return int(min(ELO_MAX, max(ELO_MIN, r or 1500)))


def decide(ch: dict, cfg: Config, active: int) -> str | None:
    """None to accept, else a Lichess decline reason."""
    c = ch.get("challenger") or {}
    tc = ch.get("timeControl") or {}
    if cfg.allow and (c.get("id") or "").lower() not in cfg.allow:
        return "later"
    if c.get("title") == "BOT":
        return "noBot"
    if (ch.get("variant") or {}).get("key") != "standard":
        return "standard"
    if tc.get("type") != "clock":
        return "timeControl"
    if tc.get("limit", 0) < MIN_BASE_S:
        return "tooFast"
    if tc.get("limit", 0) > MAX_BASE_S:
        return "tooSlow"
    if ch.get("rated") and not cfg.rated:
        return "casual"
    if active >= cfg.max_games:
        return "later"
    return None


class Lichess:
    def __init__(self, token: str, base: str = "https://lichess.org"):
        self.http = httpx.AsyncClient(base_url=base, headers={"Authorization": f"Bearer {token}"},
                                      timeout=httpx.Timeout(10.0, read=60.0))

    async def stream(self, path: str):
        """NDJSON lines; Lichess sends keep-alive newlines, so a 60 s silence means the stream is dead."""
        async with self.http.stream("GET", path) as r:
            if r.status_code == 429:
                await asyncio.sleep(60)  # Lichess: wait a full minute after a 429
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.strip():
                    yield json.loads(line)

    async def post(self, path: str, data: dict | None = None) -> httpx.Response:
        r = await self.http.post(path, data=data)
        if r.status_code == 429:
            log.warning("429 from Lichess on %s, waiting 60 s", path)
            await asyncio.sleep(60)
        return r

    async def account(self) -> dict:
        r = await self.http.get("/api/account")
        r.raise_for_status()
        return r.json()


class Infer:
    def __init__(self, url: str):
        self.http = httpx.AsyncClient(base_url=url, timeout=10.0)

    async def move(self, bot: str, elo: int, opp_elo: int, moves: list[str]) -> tuple[str, float]:
        last = None
        for attempt in range(3):
            try:
                r = await self.http.post("/api/move", json={"bot": bot, "elo": elo, "opp_elo": opp_elo, "moves": moves})
                r.raise_for_status()
                d = r.json()
                return d["move"], float(d["top"][0]["p"]) if d.get("top") else 0.5
            except httpx.HTTPError as e:
                last = e
                await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"infer unreachable: {last}")


class Game:
    """One Lichess game: follow the game stream, move on our turn after a human-like pause."""

    def __init__(self, lc: Lichess, infer: Infer, cfg: Config, me: str, game_id: str,
                 rng: np.random.Generator | None = None, sleep=asyncio.sleep, ledger: Ledger | None = None):
        self.lc, self.infer, self.cfg, self.me, self.id = lc, infer, cfg, me, game_id
        self.ledger = ledger
        self.opp_id: str | None = None
        self.plies = 0
        self.rng = rng or np.random.default_rng()
        self.sleep = sleep
        self.color: str | None = None
        self.elo = 1500
        self.inc_ms = 0
        self.moved_at_ply = -1
        self.pending: asyncio.Task | None = None
        self.greeted = False

    async def run(self) -> None:
        for _ in range(20):  # reconnect a dropped game stream
            try:
                async for ev in self.lc.stream(f"/api/bot/game/stream/{self.id}"):
                    if await self.on_event(ev):
                        return
                return  # stream ended normally: the game is over
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                log.warning("game %s stream: %s; reconnecting", self.id, e)
                await asyncio.sleep(2)

    async def on_event(self, ev: dict) -> bool:
        """Returns True when the game is over."""
        if ev.get("type") == "gameFull":
            w, b = ev["white"], ev["black"]
            self.color = "white" if (w.get("id") or "").lower() == self.me else "black"
            opp = b if self.color == "white" else w
            self.opp_id = opp.get("id")
            self.elo = clamp_elo(opp.get("rating"))
            self.inc_ms = int((ev.get("clock") or {}).get("increment", 0))
            if not self.greeted:
                self.greeted = True
                await self.lc.post(f"/api/bot/game/{self.id}/chat", {"room": "player", "text": (
                    f"Hi! I'm a doppel: a human-like chess model (Maia-3) playing at your rating, {self.elo}. "
                    "My think times are simulated. Have fun!")})
                await self.lc.post(f"/api/bot/game/{self.id}/chat", {"room": "player", "text": SOURCE_NOTE})
            state = ev["state"]
        elif ev.get("type") == "gameState":
            state = ev
        else:
            return False
        if state.get("status") != "started":
            if self.pending:
                self.pending.cancel()
            log.info("game %s over: %s %s", self.id, state.get("status"), state.get("winner"))
            if self.ledger:
                self.ledger.game_end(self.id, self.opp_id, self.elo, self.color, state.get("status"), state.get("winner"),
                                     len(state.get("moves", "").split()))
            return True
        moves = state.get("moves", "").split()
        ply = len(moves)
        if ((ply % 2 == 0) == (self.color == "white")) and ply != self.moved_at_ply:
            self.moved_at_ply = ply
            left = state.get("wtime" if self.color == "white" else "btime", 60_000)
            self.pending = asyncio.create_task(self.move(moves, int(left)))
        return False

    async def move(self, moves: list[str], left_ms: int) -> None:
        board = chess.Board()
        for u in moves:
            board.push_uci(u)
        for attempt in range(3):  # a refused move would otherwise leave the bot to flag
            t0 = time.monotonic()
            try:
                uci, p_top = await self.infer.move(self.cfg.bot, self.elo, self.elo, moves)
            except RuntimeError as e:
                log.error("game %s: %s", self.id, e)
                await self.sleep(1.0)
                continue
            if attempt == 0:
                think = heuristic_think_ms(left_ms, self.inc_ms, len(moves), board.legal_moves.count(), p_top,
                                           self.rng)
                await self.sleep(max(0.0, think / 1000 - (time.monotonic() - t0)))
            r = await self.lc.post(f"/api/bot/game/{self.id}/move/{uci}")
            if r.status_code == 200:
                return
            log.warning("game %s move %s refused: %s %s", self.id, uci, r.status_code, r.text[:200])


async def run(cfg: Config, token: str, infer_url: str) -> None:
    lc, infer = Lichess(token), Infer(infer_url)
    acct = await lc.account()
    me = acct["id"]
    if acct.get("title") != "BOT":
        raise SystemExit(f"{me} is not a BOT account")
    log.info("playing as %s with %s; allow=%s rated=%s", me, cfg.bot, sorted(cfg.allow) or "anyone", cfg.rated)
    games: dict[str, asyncio.Task] = {}
    gate, ledger = Gate(), Ledger()
    backoff = 1
    while True:
        try:
            async for ev in lc.stream("/api/stream/event"):
                backoff = 1
                games = {k: t for k, t in games.items() if not t.done()}
                kind = ev.get("type")
                if kind == "challenge":
                    ch = ev["challenge"]
                    if (ch.get("challenger") or {}).get("id") == me:
                        continue
                    who = (ch.get("challenger") or {}).get("id")
                    shed = gate.check(who or "?")
                    if shed:  # flood: don't answer at all; Lichess expires the challenge
                        ledger.shed(shed)
                        continue
                    ledger.flush_shed()
                    reason = decide(ch, cfg, len(games))
                    ledger.challenge(ch, "accept" if reason is None else reason)
                    if reason is None:
                        log.info("accept %s from %s", ch["id"], who)
                        await lc.post(f"/api/challenge/{ch['id']}/accept")
                    else:
                        log.info("decline %s from %s: %s (tc=%s rated=%s variant=%s)", ch["id"], who, reason, ch.get("timeControl"),
                                 ch.get("rated"), (ch.get("variant") or {}).get("key"))
                        await lc.post(f"/api/challenge/{ch['id']}/decline", {"reason": reason})
                elif kind == "gameStart":
                    gid = ev["game"]["gameId"]
                    if gid not in games:
                        log.info("game %s start", gid)
                        games[gid] = asyncio.create_task(Game(lc, infer, cfg, me, gid, ledger=ledger).run())
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            log.warning("event stream: %s; reconnecting in %d s", e, backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", default=str(Path.home() / ".config/doppel/lichess_bot.token"))
    ap.add_argument("--infer-url", default="http://127.0.0.1:8614")
    ap.add_argument("--bot", default="maia3-79m")
    ap.add_argument("--allow", default="", help="comma-separated Lichess usernames allowed to challenge; empty = anyone")
    ap.add_argument("--rated", action="store_true")
    ap.add_argument("--max-games", type=int, default=2)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its request lines would be noise; it never logs the token
    cfg = Config(bot=a.bot, allow={u.strip().lower() for u in a.allow.split(",") if u.strip()}, rated=a.rated,
                 max_games=a.max_games)
    token = Path(a.token_file).read_text().strip()
    asyncio.run(run(cfg, token, a.infer_url))


if __name__ == "__main__":
    main()
