"""Operator commands (Stage 0, run on the box).

    # Give every game that has no account (played before install ids existed) to one account.
    # Stage 0 has a single player, so this is how their early games become exportable once they consent.
    .venv/bin/python -m doppel_api.admin claim-unowned --user <user id from GET /api/me>
    .venv/bin/python -m doppel_api.admin users
"""

from __future__ import annotations

import argparse

from sqlalchemy import func, select, update

from .accounts import holds_consent
from .db import Game, User, make_sessionmaker


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("claim-unowned")
    c.add_argument("--user", required=True)
    sub.add_parser("users")
    a = ap.parse_args()
    sessions = make_sessionmaker()
    with sessions() as sess:
        if a.cmd == "claim-unowned":
            if sess.get(User, a.user) is None:
                raise SystemExit(f"no user {a.user}")
            n = sess.execute(update(Game).where(Game.user_id.is_(None)).values(user_id=a.user)).rowcount
            sess.commit()
            print(f"claimed {n} games for {a.user}")
        else:
            for u in sess.scalars(select(User).order_by(User.created_at)):
                n = sess.scalar(select(func.count(Game.id)).where(Game.user_id == u.id))
                print(u.id, u.display_name or "-", f"{n} games", "consent" if holds_consent(sess, u.id) else "no consent")


if __name__ == "__main__":
    main()
