# Doppel

Synthetic chess opponents with per-game state ("tilt"), a gamified client, and an open research loop. AGPL-3.0.

| Dir | What |
|---|---|
| `client/` | Flutter app (iOS, Android, web) |
| `backend/` | api, infer, lichess_bot: serves doppels on cloud CPU |
| `lab/` | research sandbox on the dev box |
| `packages/` | code shared by lab and backend |
| `contracts/` | schemas between the three |
| `demo/` | temporary PWA + move server |

Setup: `vendor/fetch.sh && uv sync`. Architecture: [ARCHITECTURE.md](ARCHITECTURE.md).
