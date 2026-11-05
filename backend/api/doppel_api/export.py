"""Native-corpus export (Contract 2, backend -> lab): one Parquet row per fully clocked, consented game.

Only games of accounts that hold training consent at export time are exported. Games played before
accounts existed (no user) are never exported. Players appear as `doppel:<h>` where h is a salted hash
with a fresh salt per export, so the lab cannot link a player across exports or back to an account.

Every exported game id goes into the `exported` ledger. When a later export finds a ledgered game whose
user withdrew consent, deleted the account, or whose game was deleted, it writes the id to
`<out stem>.tombstones.parquet` and the lab must delete that game from every copy.

Columns match lab ingest (data/human/*.parquet: moves, clocks = mover's seconds left after each move, ...),
plus per-ply timing lists. Human think times are real labels; bot rows carry think_source "heuristic_v1"
(synthetic) and must not train a think-time head.

    .venv/bin/python -m doppel_api.export --out data/native/doppel_native.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select

from .accounts import holds_consent
from .db import Exported, Game, Ply, make_sessionmaker


def run(sessions, include_active: bool = False, salt: str | None = None) -> tuple[list[dict], list[str]]:
    """Returns (rows, tombstoned game ids) and updates the ledger."""
    salt = salt or secrets.token_hex(16)
    pid = lambda uid: "doppel:" + hashlib.sha256(f"{salt}:{uid}".encode()).hexdigest()[:16]  # noqa: E731
    out, now = [], time.time()
    with sessions() as sess:
        consent = {}
        games = sess.scalars(select(Game).where(Game.user_id.is_not(None)).order_by(Game.created_at)).all()
        for g in games:
            if g.user_id not in consent:
                c = holds_consent(sess, g.user_id)
                consent[g.user_id] = c.version if c else None
            if consent[g.user_id] is None or (g.status != "over" and not include_active):
                continue
            plies = sess.scalars(select(Ply).where(Ply.game_id == g.id).order_by(Ply.ply)).all()
            if len(plies) != len(g.moves) or not plies:
                continue  # not fully clocked (pre-timing games)
            human = pid(g.user_id)
            hw = g.human_color == "white"
            out.append({
                "band": None, "game_id": g.id,
                "white": human if hw else g.package_id, "black": g.package_id if hw else human,
                "white_elo": g.human_elo if hw else g.bot_elo, "black_elo": g.bot_elo if hw else g.human_elo,
                "base": g.base_ms // 1000, "inc": g.inc_ms // 1000,
                "result": g.result, "termination": g.termination,
                "moves": [p.uci for p in plies], "clocks": [p.clock_after_ms / 1000 for p in plies],
                "source": "doppel_native", "consent_version": consent[g.user_id], "z": g.z,
                "human_color": g.human_color, "bot": g.bot, "package_id": g.package_id, "pace": g.bot_pace,
                "created_at": g.created_at, "status": g.status,
                "actor": [p.actor for p in plies], "think_source": [p.think_source for p in plies],
                "wall_ms": [p.wall_ms for p in plies], "spent_ms": [p.spent_ms for p in plies],
                "client_ms": [p.client_ms for p in plies], "premove": [p.premove for p in plies],
            })
        keep = {r["game_id"] for r in out}
        ledger = {e.game_id: e for e in sess.scalars(select(Exported))}
        tomb = [gid for gid, e in ledger.items() if gid not in keep and e.tombstoned_at is None
                and not _still_exportable_later(sess, gid, include_active)]
        for gid in tomb:
            ledger[gid].tombstoned_at = now
        for gid in keep:
            if gid in ledger:
                ledger[gid].tombstoned_at = None  # consent granted again
            else:
                sess.add(Exported(game_id=gid, exported_at=now))
        sess.commit()
    return out, tomb


def _still_exportable_later(sess, game_id: str, include_active: bool) -> bool:
    """A ledgered game missing from this export only because it is active (exported earlier with
    --include-active) is not a tombstone."""
    g = sess.get(Game, game_id)
    return g is not None and g.user_id is not None and g.status != "over" and not include_active \
        and holds_consent(sess, g.user_id) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/native/doppel_native.parquet")
    ap.add_argument("--include-active", action="store_true")
    a = ap.parse_args()
    r, tomb = run(make_sessionmaker(), a.include_active)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(r), out)
    tpath = out.with_name(out.stem + ".tombstones.parquet")
    pq.write_table(pa.table({"game_id": pa.array(tomb, pa.string())}), tpath)
    n_h = sum(x == "human" for g in r for x in g["actor"])
    print(f"wrote {out}: {len(r)} games, {n_h} human-timed moves; {len(tomb)} tombstones -> {tpath}")


if __name__ == "__main__":
    main()
