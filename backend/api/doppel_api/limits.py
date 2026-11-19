"""Request limits for the open internet: in-memory token buckets, one process (the api runs one uvicorn worker).

Keys are the client IP or the account. Behind Caddy (deploy/), Caddy sends the visitor's address in
X-Doppel-Client-IP (from CF-Connecting-IP, which it trusts only from Cloudflare's ranges); the api trusts that
header only when DOPPEL_TRUST_IP_HEADER=1, because only Caddy can reach it there. Elsewhere (Stage 0, tests) the
socket peer is the key.

Rules (rate per second, burst), chosen so a person playing never meets them and a script does:
  api_ip      every /api/* HTTP request, per IP            10/s, burst 60   (backstop)
  signup_ip   POST /api/users, per IP                      10/h, burst 10
  login_ip    /api/auth/lichess/*, per IP                  30/h, burst 20
  game        POST /api/games, per account (IP if none)    60/h, burst 10
  move        human moves and WebSocket messages, per account (IP if none)   3/s, burst 20
  import      POST /api/me/history/import, per account     6/day, burst 3
A refusal is HTTP 429 with Retry-After (seconds), or a WebSocket {"type":"error","status":429}.
DOPPEL_LIMITS=off disables every rule (the test suite does; test_limits.py turns them back on).
"""

from __future__ import annotations

import math
import os
import time

from fastapi import HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse

H, DAY = 3600.0, 86400.0
RULES: dict[str, tuple[float, float]] = {
    "api_ip": (10.0, 60),
    "signup_ip": (10 / H, 10),
    "login_ip": (30 / H, 20),
    "game": (60 / H, 10),
    "move": (3.0, 20),
    "import": (6 / DAY, 3),
}
PRUNE_AT = 50_000  # buckets kept before idle (full) ones are dropped


class Limiter:
    def __init__(self, rules: dict[str, tuple[float, float]] | None = None, enabled: bool = True,
                 clock=time.monotonic):
        self.rules, self.enabled, self.clock = dict(rules or RULES), enabled, clock
        self.b: dict[tuple[str, str], tuple[float, float]] = {}  # (rule, key) -> (tokens, at)

    @classmethod
    def from_env(cls) -> "Limiter":
        return cls(enabled=os.environ.get("DOPPEL_LIMITS", "on").lower() not in ("off", "0", "false"))

    def wait(self, rule: str, key: str) -> float:
        """Take one token; 0.0 if allowed, else the seconds until one is free (nothing is taken)."""
        if not self.enabled:
            return 0.0
        rate, burst = self.rules[rule]
        now = self.clock()
        tok, at = self.b.get((rule, key), (burst, now))
        tok = min(burst, tok + (now - at) * rate)
        if tok < 1:
            self.b[(rule, key)] = (tok, now)
            return (1 - tok) / rate
        self.b[(rule, key)] = (tok - 1, now)
        if len(self.b) > PRUNE_AT:
            self._prune(now)
        return 0.0

    def check(self, rule: str, key: str) -> None:
        """wait() as an HTTPException(429) with Retry-After."""
        w = self.wait(rule, key)
        if w:
            raise HTTPException(429, "too many requests; slow down", headers={"Retry-After": str(math.ceil(w))})

    def _prune(self, now: float) -> None:
        self.b = {k: (t, a) for k, (t, a) in self.b.items()
                  if t + (now - a) * self.rules[k[0]][0] < self.rules[k[0]][1]}


def client_ip(conn: Request | WebSocket) -> str:
    if os.environ.get("DOPPEL_TRUST_IP_HEADER") == "1":
        ip = conn.headers.get("x-doppel-client-ip")
        if ip:
            return ip.strip()
    return conn.client.host if conn.client else "unknown"


def actor(conn: Request | WebSocket, user) -> str:
    """The key for per-account rules: the account id, or the IP for anonymous requests."""
    return f"u:{user.id}" if user is not None else f"ip:{client_ip(conn)}"


PATH_RULES = (("POST", "/api/users", "signup_ip"), (None, "/api/auth/lichess/", "login_ip"))


async def middleware(request: Request, call_next):
    """Per-IP rules for every /api/* HTTP request (WebSockets are limited per message in app.game_ws)."""
    lim: Limiter | None = getattr(request.app.state, "limits", None)
    path = request.url.path
    if lim is not None and path.startswith("/api/"):
        ip = client_ip(request)
        rules = ["api_ip"] + [r for m, p, r in PATH_RULES
                              if (m is None or m == request.method) and (path == p or (p.endswith("/") and path.startswith(p)))]
        for r in rules:
            w = lim.wait(r, ip)
            if w:
                return JSONResponse({"detail": "too many requests; slow down"}, 429,
                                    headers={"Retry-After": str(math.ceil(w))})
    return await call_next(request)
