"""Recover player names and start times for the Allie 2022 blitz test set (network, ~3 min, no GPU).

    python -m doppel_lab.allie_players            # -> data/allie/players.parquet

The test file has Lichess game IDs but no usernames or dates. Lichess's public export-by-IDs endpoint
(POST /api/games/export/_ids, up to 300 IDs per request) returns both. Requests go one at a time with a pause,
and a 429 waits a minute, as the API asks. Columns: game_id, white, black (lowercase Lichess ids; null for an
anonymous or missing side), white_name, black_name, created_at (ms UTC), white_rating, black_rating, speed,
status. Games the API no longer returns are listed in the summary.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/allie/2022-test-annotated.jsonl"
OUT = ROOT / "data/allie/players.parquet"
URL = "https://lichess.org/api/games/export/_ids?moves=false&clocks=false&opening=false"
UA = "doppel-research-client"
BATCH = 300


def fetch(ids: list[str]) -> list[dict]:
    req = urllib.request.Request(URL, data=",".join(ids).encode(), method="POST",
                                 headers={"Accept": "application/x-ndjson", "User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return [json.loads(l) for l in r.read().decode().splitlines() if l.strip()]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print("429: waiting 60 s", flush=True)
                time.sleep(60)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"network error {e}; retry in 20 s", flush=True)
            time.sleep(20)
    raise RuntimeError("giving up after retries")


def side(g: dict, col: str) -> tuple[str | None, str | None, int | None]:
    p = g.get("players", {}).get(col, {})
    u = p.get("user") or {}
    return u.get("id"), u.get("name"), p.get("rating")


def main():
    ids = [json.loads(l)["game-id"].rsplit("/", 1)[-1] for l in open(SRC)]
    rows = []
    for i in range(0, len(ids), BATCH):
        for g in fetch(ids[i:i + BATCH]):
            w, b = side(g, "white"), side(g, "black")
            rows.append({"game_id": g["id"], "white": w[0], "black": b[0], "white_name": w[1], "black_name": b[1],
                         "created_at": g.get("createdAt"), "white_rating": w[2], "black_rating": b[2],
                         "speed": g.get("speed"), "status": g.get("status")})
        print(f"{min(i + BATCH, len(ids)):,} / {len(ids):,} requested, {len(rows):,} returned", flush=True)
        time.sleep(1.0)
    df = pl.DataFrame(rows)
    df.write_parquet(OUT)
    missing = sorted(set(ids) - set(df["game_id"].to_list()))
    sides = pl.concat([df["white"], df["black"]])
    print(json.dumps({"games": len(ids), "returned": df.height, "missing": len(missing), "missing_ids": missing[:20],
                      "sides_named": int(sides.is_not_null().sum()), "players": int(sides.drop_nulls().n_unique()),
                      "date_min": str(pl.from_epoch(df["created_at"], "ms").min()),
                      "date_max": str(pl.from_epoch(df["created_at"], "ms").max())}, indent=1))


if __name__ == "__main__":
    main()
