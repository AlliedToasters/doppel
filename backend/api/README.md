# backend/api

Server-authoritative Doppel-hosted games. The server holds the moves, the clocks and the doppel's per-game z
(sampled once at creation, per the Phase 1 result); clients send intents and render returned state.
Bot moves come from `backend/infer` over HTTP. Accounts, consent, ratings and the consented export are below.

| Endpoint | Does |
|---|---|
| `POST /api/games {bot, bot_elo, human_elo, color, base_s, inc_s}` | new game; the bot moves first if you are Black |
| `GET /api/games/{id}` | state (live clocks; a flag is applied on read) |
| `POST /api/games/{id}/move {uci, ply?}` | your move, then the bot's reply; `ply` refuses stale/duplicate sends |
| `POST /api/games/{id}/resign` | resign |
| `WS /api/games/{id}/ws?token=` | intents `move` / `resign` / `pace` / `sync` in; `state` after every change out |
| `POST /api/users {install_id?, display_name?}` | anonymous account; returns the bearer token once; claims the install's earlier games |
| `GET` / `PATCH` / `DELETE /api/me` | profile with consent and rating; rename; delete the account and its games |
| `GET /api/consent`, `PUT /api/me/consent {training, version}` | consent text; grant (must match the current version) or withdraw |
| `GET /api/me/games` | the account's games, newest first |

Accounts (`accounts.py`): send `Authorization: Bearer <token>`. A game made with a token can only be read or
played with it (403 otherwise; the WebSocket closes with 4403). Anonymous games still work until
`DOPPEL_REQUIRE_AUTH=1`.

Ratings (`ratings.py`): MAP Bradley-Terry on the Elo scale over all rated games (over, with an account, both
sides moved twice). Each `package@bot_elo` is an opponent anchored at its Elo setting (sd 100); users start
at 1500 (sd 350). Refit after every rated game; `python -m doppel_api.ratings` refits and prints.

Rules: clocks start once both sides have moved; increment is added after each charged move; flagging
against insufficient material is a draw.

Native corpus: every move is stored in `plies` with its timing (wall, charged, clock after, device think time,
premove flag). Human times are real labels; bot times are `heuristic_v1` (synthetic) and must never train a
think-time head. Export to the lab's game format (consenting accounts only, player ids salted fresh per
export, tombstones for games withdrawn or deleted since an earlier export, in `<stem>.tombstones.parquet`):

    .venv/bin/python -m doppel_api.export --out data/native/doppel_native.parquet
    .venv/bin/python -m doppel_api.admin users                       # accounts, game counts, consent
    .venv/bin/python -m doppel_api.admin claim-unowned --user <id>   # Stage 0: pre-install-id games to one account

Storage: SQLite at `data/api.db` (Stage 0); `DOPPEL_DB_URL=postgresql+psycopg://...` with the `postgres` extra for Stage 1.

Stage 0 (bigbox): the api is the only service the phone sees (tailscale serve :8460 -> 127.0.0.1:8612). It
also passes `/api/bots` and the stateless `/api/move` through to infer and serves the APK from `demo/static`.

Runs as the user unit `doppel-api` (~/.config/systemd/user/doppel-api.service; `systemctl --user restart doppel-api`, logs in logs/). The command below is what the unit runs.

    DOPPEL_INFER_URL=http://127.0.0.1:8614 DOPPEL_STATIC=demo/static \
      .venv/bin/uvicorn doppel_api.app:app --host 127.0.0.1 --port 8612
    .venv/bin/python -m pytest -q backend/api/tests
