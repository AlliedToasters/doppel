"""Native-corpus export (Contract 2, backend -> lab): one Parquet row per fully clocked game.

Columns match lab ingest (data/human/*.parquet: moves, clocks = mover's seconds left after each move, ...),
plus per-ply timing lists. Human think times are real labels; bot rows carry think_source "heuristic_v1"
(synthetic) and must not train a think-time head. Games recorded before per-ply timing existed are skipped.

    .venv/bin/python -m doppel_api.export --out data/native/doppel_native.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select

from .db import Game, Ply, make_sessionmaker


def rows(sessions, include_active: bool = False) -> list[dict]:
    out = []
    with sessions() as sess:
        games = sess.scalars(select(Game).order_by(Game.created_at)).all()
        for g in games:
            if g.status != "over" and not include_active:
                continue
            plies = sess.scalars(select(Ply).where(Ply.game_id == g.id).order_by(Ply.ply)).all()
            if len(plies) != len(g.moves) or not plies:
                continue  # not fully clocked (pre-timing games)
            human = f"doppel:{g.player or 'anon'}"
            hw = g.human_color == "white"
            out.append({
                "band": None, "game_id": g.id,
                "white": human if hw else g.package_id, "black": g.package_id if hw else human,
                "white_elo": g.human_elo if hw else g.bot_elo, "black_elo": g.bot_elo if hw else g.human_elo,
                "base": g.base_ms // 1000, "inc": g.inc_ms // 1000,
                "result": g.result, "termination": g.termination,
                "moves": [p.uci for p in plies], "clocks": [p.clock_after_ms / 1000 for p in plies],
                "source": "doppel_native",
                "human_color": g.human_color, "bot": g.bot, "package_id": g.package_id, "pace": g.bot_pace,
                "created_at": g.created_at, "status": g.status,
                "actor": [p.actor for p in plies], "think_source": [p.think_source for p in plies],
                "wall_ms": [p.wall_ms for p in plies], "spent_ms": [p.spent_ms for p in plies],
                "client_ms": [p.client_ms for p in plies], "premove": [p.premove for p in plies],
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/native/doppel_native.parquet")
    ap.add_argument("--include-active", action="store_true")
    a = ap.parse_args()
    r = rows(make_sessionmaker(), a.include_active)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(r), a.out)
    n_h = sum(x == "human" for g in r for x in g["actor"])
    print(f"wrote {a.out}: {len(r)} games, {n_h} human-timed moves")


if __name__ == "__main__":
    main()
