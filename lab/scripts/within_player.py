"""Is human error persistence a per-game state or a per-player trait?

For players with >= 8 games, compare
  within-game  r( q_h1[g], q_h2[g] )           same player, same game
  cross-game   r( q_h1[g], q_h2[g'] ), g != g'  same player, different games (player trait only)
  residual     r of leave-one-out player-demeaned halves (per-game covariance beyond the trait)
where q = log1p(ACPL) of that player's half of the game. Bootstrap over players.
Run: PYTHONPATH=. .venv/bin/python scripts/within_player.py
"""

import json
from collections import defaultdict

import numpy as np
import polars as pl

from doppel_lab.metrics import game_stats

MIN_GAMES = 8
games = pl.read_parquet("data/human/repeat_2026-10.parquet")
from pathlib import Path
ev = pl.concat([pl.read_parquet(f) for f in ("data/evals/human.parquet", "data/evals/human_repeat.parquet") if Path(f).exists()])
ev = dict(zip(ev["game_id"], ev["evals"]))

rows = defaultdict(list)  # player -> [(h1, h2, bl_h1, bl_h2, tc, band)]
for r in games.iter_rows(named=True):
    if r["game_id"] not in ev:
        continue
    s = game_stats(r, ev[r["game_id"]])
    if len(s["players"]) != 2:
        continue
    for side, name in enumerate((r["white"].lower(), r["black"].lower())):
        p = s["players"][side]
        rows[name].append((np.log1p(p["acpl_h1"]), np.log1p(p["acpl_h2"]), p["bl_h1"], p["bl_h2"],
                           f'{r["base"]}+{r["inc"]}', r["band"]))
players = {k: np.array([v[:4] for v in vs]) for k, vs in rows.items() if len(vs) >= MIN_GAMES}
tcs = {k: [v[4] for v in rows[k]] for k in players}
print(f"{len(players)} players, {sum(len(v) for v in players.values())} player-games")

# time-control residualisation: subtract the global mean of each half per time control
allv = [(tc, x) for k in players for tc, x in zip(tcs[k], players[k])]
tc_mean = defaultdict(list)
for tc, x in allv:
    tc_mean[tc].append(x)
tc_mean = {tc: np.mean(v, axis=0) for tc, v in tc_mean.items()}
players_tc = {k: players[k] - np.array([tc_mean[t] for t in tcs[k]]) for k in players}


def stats(P, keys, a, b):
    wg, cg, res = [], [], []
    for k in keys:
        X = P[k]
        x, y = X[:, a], X[:, b]
        n = len(x)
        wg.append(np.c_[x, y])
        # cross-game: pair each game's h1 with the next game's h2 (cyclic), same player
        cg.append(np.c_[x, np.roll(y, 1)])
        lx = (x.sum() - x) / (n - 1)
        ly = (y.sum() - y) / (n - 1)
        res.append(np.c_[x - lx, y - ly])
    f = lambda L: float(np.corrcoef(np.concatenate(L).T)[0, 1])
    return f(wg), f(cg), f(res)


rng = np.random.default_rng(0)
keys = list(players)
out = {}
for label, P in (("raw", players), ("tc_resid", players_tc)):
    for metric, a, b in (("acpl", 0, 1), ("blunder_rate", 2, 3)):
        point = stats(P, keys, a, b)
        boots = np.array([stats(P, rng.choice(keys, len(keys)), a, b) for _ in range(300)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        share = (boots[:, 0] - boots[:, 1]) / boots[:, 0]
        out[f"{label}/{metric}"] = {
            n: {"r": round(point[i], 4), "ci": [round(lo[i], 4), round(hi[i], 4)]}
            for i, n in enumerate(("within_game", "cross_game", "residual"))
        } | {"per_game_share": [round(float(np.percentile(share, q)), 3) for q in (2.5, 50, 97.5)]}
        print(label, metric, json.dumps(out[f"{label}/{metric}"]))
out["n_players"], out["n_player_games"] = len(players), int(sum(len(v) for v in players.values()))
json.dump(out, open("lab/results/within_player_2026-10.json", "w"), indent=1)
