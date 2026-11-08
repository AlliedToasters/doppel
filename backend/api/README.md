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
| `GET /api/consent`, `PUT /api/me/consent {training, version}` | consent texts (`training`/`version`, `history`/`history_version`); grant training (must match the current version) or withdraw |
| `PUT /api/me/history-consent {granted, version, adult}` | grant history-v1 (needs Lichess sign-in, not kid mode, `adult: true` = the 18+ box) or withdraw it, which deletes the imported games and the import status |
| `POST /api/me/history/import` | queue an import of your public Lichess games (202); 403 with a plain `detail` if not allowed |
| `GET /api/me/history` | import status: `state` (none, queued, running, done, error), `games` stored, `added` by the latest run, `error`, `updated_at`, `max_games`, `can_import`, `reason` |
| `GET /api/me/games` | the account's games, newest first |
| `GET /api/auth/lichess/start?client=web\|app` | Sign in with Lichess (PKCE, scope `preference:read`, only for kid mode): 302 to lichess.org |
| `GET /api/auth/lichess/callback` | reads the Lichess id and kid mode once, revokes the Lichess token, redirects to the client with a one-time `login_code` |
| `POST /api/auth/lichess/exchange {code}` | login code -> Doppel token for this device; links or merges the caller's anonymous account (`lichess_login.py`) |

Accounts (`accounts.py`): send `Authorization: Bearer <token>`. A game made with a token can only be read or
played with it (403 otherwise; the WebSocket closes with 4403). Anonymous games still work until
`DOPPEL_REQUIRE_AUTH=1`. Sign-in redirects are built from `DOPPEL_PUBLIC_URL` (the URL clients reach the api at).

Lichess history (`history.py`): an opt-in, separate from the training consent, to import the user's public
Lichess games for their personal doppel (and for general models only alongside the training consent; the export
doesn't include them yet). Allowed for accounts signed in with Lichess that are not in Lichess kid mode and
whose history-v1 consent carries the self-declared 18+ flag. Kid mode is read once at sign-in
(`GET /api/account/kid`, scope `preference:read`) before the Lichess token is revoked; accounts that signed in
before this need to sign in again. The import streams the public export
(`GET /api/games/user/{id}`, ndjson, `clocks`, `opening`, standard perf types only, newest first) with no token,
runs as a background task in the api process, one export stream at a time across the server, and waits at least
60 s after a 429 (or `Retry-After`, if longer). It is incremental (`since` = newest stored game after the last
completed run; duplicates are skipped) and capped per run by `DOPPEL_HISTORY_MAX_GAMES` (default 5000).
Rows: `imported_games` (game id unique per user, played_at, speed/perf, rated, colour, both ratings, result,
termination, time control, SAN moves, clocks in centiseconds, opening; the opponent's name is not stored) and
`history_imports` (one status row per user). A run cut short by a restart shows as an error; import again.

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
New tables are created at startup; new columns are added by the additive list in `db.py` (nullable, no default).

Environment: `DOPPEL_DB_URL`, `DOPPEL_INFER_URL`, `DOPPEL_STATIC`, `DOPPEL_PUBLIC_URL` (sign-in redirects),
`DOPPEL_LICHESS_URL` (default https://lichess.org), `DOPPEL_LICHESS_CLIENT_ID` (default doppelbot.org),
`DOPPEL_REQUIRE_AUTH` (1 refuses anonymous games), `DOPPEL_HISTORY_MAX_GAMES` (games per import run, default 5000).

Stage 0 (bigbox): the api is the only service the phone sees (tailscale serve :8460 -> 127.0.0.1:8612). It
also passes `/api/bots` through to infer and serves the APK from `demo/static`. There is deliberately no stateless
`/api/move`: bot moves exist only inside server-hosted games, so the api is never a move-suggestion service for a
live Lichess game. Bearer tokens are redacted from uvicorn's log lines (`?token=REDACTED`).

Runs as the user unit `doppel-api` (~/.config/systemd/user/doppel-api.service; `systemctl --user restart doppel-api`, logs in logs/). The command below is what the unit runs.

    DOPPEL_INFER_URL=http://127.0.0.1:8614 DOPPEL_STATIC=demo/static \
      .venv/bin/uvicorn doppel_api.app:app --host 127.0.0.1 --port 8612
    .venv/bin/python -m pytest -q backend/api/tests
