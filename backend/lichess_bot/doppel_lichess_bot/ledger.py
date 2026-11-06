"""Challenge ledger and flood gate for doppelbot.

Every challenge (with its decision) and every finished game is appended to data/lichess_bot/ledger.jsonl.
The gate sheds load during a flood: past a global challenge rate, or when one account keeps challenging, the
runner stops answering (no accept/decline request), so a burst costs nothing on our side and cannot get the
bot's token rate-limited by Lichess. Lichess expires unanswered challenges on its own. While shedding, the
ledger writes one summary line per minute instead of one line per challenge, so the file cannot balloon.

    .venv/bin/python -m doppel_lichess_bot.ledger            # stats for the last 7 days
    .venv/bin/python -m doppel_lichess_bot.ledger --days 1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

log = logging.getLogger("doppel.lichess_bot")
ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "data/lichess_bot/ledger.jsonl"
BREAKER = ROOT / "data/lichess_bot/BREAKER_TRIPPED"  # present = hard off until a human deletes it


class Gate:
    """Returns None (handle normally), "flood" (too many challenges overall) or "spam" (one account)."""

    def __init__(self, global_per_min: int = 20, user_per_10min: int = 5, user_block_s: int = 3600,
                 trip_per_10min: int = 100, clock=time.time):
        self.global_per_min, self.user_per_10min, self.user_block_s = global_per_min, user_per_10min, user_block_s
        self.trip_per_10min = trip_per_10min
        self.clock = clock
        self.recent: deque[float] = deque()
        self.window: deque[float] = deque()  # every challenge in the last 10 min, for the hard breaker
        self.by_user: dict[str, deque[float]] = defaultdict(deque)
        self.blocked: dict[str, float] = {}
        self.flooding = False

    def check(self, who: str) -> str | None:
        t = self.clock()
        while self.recent and self.recent[0] < t - 60:
            self.recent.popleft()
        q = self.by_user[who]
        while q and q[0] < t - 600:
            q.popleft()
        while self.window and self.window[0] < t - 600:
            self.window.popleft()
        self.recent.append(t)
        self.window.append(t)
        q.append(t)
        if self.blocked.get(who, 0) > t:
            return "spam"
        if len(q) > self.user_per_10min:
            self.blocked[who] = t + self.user_block_s
            log.warning("flood gate: %s sent %d challenges in 10 min; ignoring them for %d s", who, len(q),
                        self.user_block_s)
            return "spam"
        flooding = len(self.recent) > self.global_per_min
        if flooding != self.flooding:
            self.flooding = flooding
            log.warning("flood gate: %s (%d challenges in the last minute)", "SHEDDING" if flooding else "recovered",
                        len(self.recent))
        return "flood" if flooding else None

    @property
    def tripped(self) -> bool:
        """Hard circuit breaker: a sustained flood (shedding alone keeps up with bursts, not with this)."""
        return len(self.window) > self.trip_per_10min


def trip_breaker(n: int, path: Path = BREAKER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"tripped {time.strftime('%Y-%m-%d %H:%M:%S')}: {n} challenges in 10 min. "
                    "Delete this file to resume (the runner checks every 30 s).\n")


class Ledger:
    def __init__(self, path: Path = LEDGER, clock=time.time):
        self.path, self.clock = path, clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._shed: Counter = Counter()
        self._shed_since: float | None = None

    def _write(self, row: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    def challenge(self, ch: dict, decision: str) -> None:
        c = ch.get("challenger") or {}
        tc = ch.get("timeControl") or {}
        self._write({"ts": round(self.clock(), 3), "kind": "challenge", "id": ch.get("id"), "challenger": c.get("id"),
                     "title": c.get("title"), "rating": c.get("rating"), "rated": ch.get("rated"),
                     "variant": (ch.get("variant") or {}).get("key"), "tc": tc.get("show") or tc.get("type"),
                     "decision": decision})

    def shed(self, why: str) -> None:
        """Counted, not written per challenge; flushed at most once a minute."""
        t = self.clock()
        self._shed[why] += 1
        if self._shed_since is None:
            self._shed_since = t
        if t - self._shed_since >= 60:
            self.flush_shed()

    def flush_shed(self) -> None:
        if self._shed:
            self._write({"ts": round(self.clock(), 3), "kind": "shed", "since": round(self._shed_since or 0, 3),
                         **dict(self._shed)})
        self._shed.clear()
        self._shed_since = None

    def game_end(self, game_id: str, opponent: str | None, elo: int, color: str | None, status: str | None,
                 winner: str | None, plies: int) -> None:
        self._write({"ts": round(self.clock(), 3), "kind": "game", "id": game_id, "opponent": opponent, "elo": elo,
                     "color": color, "status": status, "winner": winner, "plies": plies})


def stats(rows: list[dict], since: float) -> str:
    rows = [r for r in rows if r.get("ts", 0) >= since]
    ch = [r for r in rows if r["kind"] == "challenge"]
    games = [r for r in rows if r["kind"] == "game"]
    shed = [r for r in rows if r["kind"] == "shed"]
    per_min = Counter(int(r["ts"] // 60) for r in ch)
    per_day = Counter(time.strftime("%Y-%m-%d", time.localtime(r["ts"])) for r in ch)
    dec = Counter(r["decision"] for r in ch)
    who = Counter(r["challenger"] for r in ch)
    res = Counter("win" if g["winner"] == g["color"] else "draw" if g["winner"] is None else "loss" for g in games)
    n_shed = sum(v for r in shed for k, v in r.items() if k in ("flood", "spam"))
    lines = [
        f"challenges: {len(ch)} from {len(who)} accounts; peak {max(per_min.values(), default=0)}/min",
        "per day: " + (", ".join(f"{d} {n}" for d, n in sorted(per_day.items())) or "none"),
        "decisions: " + (", ".join(f"{k} {v}" for k, v in dec.most_common()) or "none"),
        "top challengers: " + (", ".join(f"{k} {v}" for k, v in who.most_common(5)) or "none"),
        f"games: {len(games)} (doppelbot {res['win']}W {res['draw']}D {res['loss']}L)",
        f"shed (flood gate): {n_shed} challenges in {len(shed)} episodes" if shed else "shed (flood gate): none",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--path", default=str(LEDGER))
    a = ap.parse_args()
    p = Path(a.path)
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []
    print(stats(rows, time.time() - a.days * 86400))


if __name__ == "__main__":
    main()
