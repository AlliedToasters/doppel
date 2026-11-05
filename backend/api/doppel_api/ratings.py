"""Ratings: a MAP Bradley-Terry fit (Elo scale) over every rated game, users and bots together.

Each opponent is a package at an Elo setting ("bot:<package>@<elo>"), anchored by a tight prior at that
setting, so bots stay on the Lichess scale their conditioning targets and users are measured against them.
Users start from a wide prior at 1500. Draws score 0.5. rd is the posterior standard deviation (Laplace).

A game is rated when it is over with a result, has an account, and both sides made at least two moves.
Stage 0 refits everything after each finished game (a few ms at this size); make it incremental before
the user count makes that slow.

    .venv/bin/python -m doppel_api.ratings          # refit and print
"""

from __future__ import annotations

import math
import time

from sqlalchemy import delete, select

from .db import Game, Rating, make_sessionmaker

K = math.log(10) / 400
USER_PRIOR = (1500.0, 350.0)
BOT_SD = 100.0
MIN_PLIES = 4


def bot_subject(g: Game) -> str:
    return f"bot:{g.package_id}@{g.bot_elo}"


def rated(g: Game) -> bool:
    return g.status == "over" and g.result in ("1-0", "0-1", "1/2-1/2") and g.user_id is not None \
        and len(g.moves) >= MIN_PLIES and g.termination != "abort"


def user_score(g: Game) -> float:
    if g.result == "1/2-1/2":
        return 0.5
    return 1.0 if (g.result == "1-0") == (g.human_color == "white") else 0.0


def fit(results: list[tuple[str, str, float]], priors: dict[str, tuple[float, float]], iters: int = 100,
        tol: float = 1e-3) -> dict[str, tuple[float, float, int]]:
    """results: (a, b, score of a). priors: subject -> (mean, sd); every subject in results needs one.
    Coordinate-wise Newton on the log posterior; returns subject -> (rating, rd, games)."""
    r = {s: mu for s, (mu, _) in priors.items()}
    games: dict[str, list[tuple[str, float]]] = {s: [] for s in priors}
    for a, b, s in results:
        games[a].append((b, s))
        games[b].append((a, 1.0 - s))
    for _ in range(iters):
        moved = 0.0
        for s, gl in games.items():
            mu, sd = priors[s]
            g1, h = -(r[s] - mu) / sd**2, 1 / sd**2
            for o, sc in gl:
                p = 1 / (1 + math.exp(-K * (r[s] - r[o])))
                g1 += K * (sc - p)
                h += K * K * p * (1 - p)
            step = g1 / h
            r[s] += step
            moved = max(moved, abs(step))
        if moved < tol:
            break
    out = {}
    for s, gl in games.items():
        h = 1 / priors[s][1] ** 2
        for o, _ in gl:
            p = 1 / (1 + math.exp(-K * (r[s] - r[o])))
            h += K * K * p * (1 - p)
        out[s] = (r[s], 1 / math.sqrt(h), len(gl))
    return out


def refit(sessions) -> dict[str, tuple[float, float, int]]:
    with sessions() as sess:
        games = [g for g in sess.scalars(select(Game).where(Game.status == "over", Game.user_id.is_not(None)))
                 if rated(g)]
        priors: dict[str, tuple[float, float]] = {}
        results = []
        for g in games:
            u, b = f"user:{g.user_id}", bot_subject(g)
            priors.setdefault(u, USER_PRIOR)
            priors.setdefault(b, (float(g.bot_elo), BOT_SD))
            results.append((u, b, user_score(g)))
        out = fit(results, priors)
        now = time.time()
        sess.execute(delete(Rating))
        sess.add_all(Rating(subject=s, rating=round(v[0], 1), rd=round(v[1], 1), games=v[2], updated_at=now)
                     for s, v in out.items())
        sess.commit()
    return out


if __name__ == "__main__":
    for s, (r, rd, n) in sorted(refit(make_sessionmaker()).items(), key=lambda kv: -kv[1][0]):
        print(f"{r:7.1f} ±{rd:5.1f}  {n:4d} games  {s}")
