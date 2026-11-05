# backend/api

Server-authoritative Doppel-hosted games. The server holds the moves, the clocks and the doppel's per-game z
(sampled once at creation, per the Phase 1 result); clients send intents and render returned state.
Bot moves come from `backend/infer` over HTTP. Later: accounts, consent, ratings (Bradley-Terry), export.

| Endpoint | Does |
|---|---|
| `POST /api/games {bot, bot_elo, human_elo, color, base_s, inc_s}` | new game; the bot moves first if you are Black |
| `GET /api/games/{id}` | state (live clocks; a flag is applied on read) |
| `POST /api/games/{id}/move {uci, ply?}` | your move, then the bot's reply; `ply` refuses stale/duplicate sends |
| `POST /api/games/{id}/resign` | resign |
| `WS /api/games/{id}/ws` | intents `move` / `resign` / `sync` in; `state` after every change out |

Rules: clocks start once both sides have moved; increment is added after each charged move; flagging
against insufficient material is a draw.

Storage: SQLite at `data/api.db` (Stage 0); `DOPPEL_DB_URL=postgresql+psycopg://...` with the `postgres` extra for Stage 1.

    .venv/bin/uvicorn doppel_api.app:app --host 127.0.0.1 --port 8613   # needs infer on :8612
    .venv/bin/python -m pytest -q backend/api/tests
