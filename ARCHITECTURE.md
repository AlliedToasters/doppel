# Doppel codebase architecture (proposal, 2026-11-04)

Three deployables in one monorepo, connected by two versioned contracts:

```
            model package (ONNX + manifest)              OpenAPI + WebSocket schema
   lab  ──────────────────────────────────▶  backend  ◀──────────────────────────────▶  client
(this box)                                  (cloud CPU)                              (iOS / Android / web)
   ▲                                             │
   └──────── consented game export (Parquet) ────┘
```

- **lab** trains and evaluates. It never serves users.
- **backend** serves doppels and owns all user data. It never trains.
- **client** shows the board and never runs a model during a Lichess-hosted game.

The model package and the game export are the only things that cross the lab↔backend boundary. Each has a schema in `contracts/`.

## Repo layout

```
doppel/
  client/                 Flutter app: iOS, Android, web
  backend/
    api/                  FastAPI: accounts, consent, Doppel-hosted games (WebSocket), ratings, export
    infer/                stateless CPU model server (ONNX Runtime), one process per model package
    lichess_bot/          one runner for the BOT account doppelbot (Lichess Bot API stream → infer)
  lab/                    research sandbox: today's doppel/ package, scripts/, results/
  packages/
    doppel_model/         shared Python: board/history encoding, move decoding, z sampling, package loader
  contracts/
    openapi.json          generated from backend/api; the client codegens from it
    model_package.schema.json
    game_export.schema.json
  deploy/                 docker compose for one VM to start; Caddy for TLS
  vendor/                 gitignored; fetch.sh pins maia3 (AGPL-3.0) and chessmimic (PolyForm Noncommercial, lab only)
  data/                   gitignored; Lichess dumps, evals, synthetic games (lab reads it from the repo root)
  demo/                   temporary PWA + move server; deleted once client/ and backend/infer serve Maia-3
  docs/                   design notes; SCIENCE.md is the canonical science program
```

Python uses one uv workspace with members `lab`, `packages/doppel_model`, `backend/api`, `backend/infer` and `backend/lichess_bot`.

- `lab` depends on torch and CUDA.
- `backend/infer` depends on `onnxruntime` and `doppel_model` only. That keeps the serving image small, with no torch.

## Contract 1: the model package (lab → backend)

```
models/<name>-<sha8>/      (gitignored locally; object storage later)
  model.onnx              teacher + residual g(board, z), exported from the lab
  manifest.json           id, sha256, input spec, Elo/band range, z prior, clock head?, lab commit, eval scores
```

- The package id is the content hash, which settles "each weights version counts as a separate player" in the rating system. A doppel identity (name, avatar, band) points at one package id. Retraining creates a new player and leaves the old one as a frozen anchor.
- `doppel_model` holds the encoding and decoding. The lab trains through it and `infer` serves through it, so the features can't drift apart between training and serving.
- A parity test runs in CI: lab torch outputs must match infer ONNX outputs on a fixed set of 1,000 positions, to 1e-4.
- The lab publishes packages to object storage, and the backend pulls them by id.

## Contract 2: the game export (backend → lab)

- A nightly job writes Parquet with the same columns as `lab/` ingest: moves, clocks, Elo, the doppel's package id and its per-game z.
- It exports only games whose players hold the training consent.
- Withdrawn consents go out as tombstones, so lab copies get deleted too.
- Player ids are salted per export. The lab never sees Lichess usernames for Doppel-hosted games.

## backend

- **api:** FastAPI on Postgres.
  - Doppel-hosted games are **server-authoritative**: the server holds the moves, the clocks and the doppel's z, and the client sends intents over a WebSocket. That's what makes ratings trustworthy and pause/resume correct: the doppel resumes in the same tilt because z lives in the game row.
  - The server also owns ratings (Bradley-Terry batch job) and consent records.
- **infer:** a stateless HTTP service.
  - Input is a position plus history, clocks, Elo and z. Output is a move distribution, a sampled move and a think time.
  - It scales horizontally, one process per CPU core group.
  - Latency budget: p95 under 150 ms per move on a 4-vCPU box, to be measured on Maia-3 79M before committing to an instance size.
- **lichess_bot:** a long-lived worker for one BOT account (`doppelbot`, upgraded 2026-11-06) that streams Lichess events and calls infer. Every doppel plays from this one account, at the challenger's rating; a second BOT account needs Lichess's OK first (their multi-account rule). Open to anyone since 2026-11-06, once the repo went public (Maia-3 is AGPL-3.0; the bot links the source in every game's chat). Casual only, at most 2 games at once. It has no database access beyond recording that a game happened; Lichess already publishes those games.
- **Stage 0 (now): bigbox over the tailnet.** Everything runs on this box, CPU only. The backend binds to 127.0.0.1, and `tailscale serve` publishes it on HTTPS at `https://<host>.<tailnet>.ts.net:<port>`, so the phone gets a real certificate and Android needs no cleartext exception. Only devices on the tailnet can reach it, so there's no public exposure, no accounts and no consent flow yet. The client reads its backend URL from `--dart-define=BACKEND_URL=...`, which means moving to the cloud later is a rebuild, not a code change. The APK is built on bigbox and installed on the Pixel 8, either by downloading it from a tailnet URL or by `adb` wireless debugging. Stage 0 lasts until real Doppel models are ready.
- **Stage 1: cloud.** One CPU VM running `api + infer + lichess_bot + postgres + caddy` under docker compose. Split infer onto its own VMs once it's the bottleneck.

## client

- **Flutter, with our own board.** One codebase builds iOS, Android and a web build that replaces today's `demo/` PWA. Flutter draws every pixel itself, so the board, drag, animation, haptics and sound are fully ours to tune. We use `dartchess` (GPL-3.0) for move rules.
- **Board skins behind one interface.** Everything above the board (game state, clocks, the backend connection, tilt indicators, menus) talks to a `BoardSurface` interface: show a position, emit a move intent, show premoves, highlight, flip. A skin is one implementation of that interface plus a theme.
  - **Classic skin:** wraps Lichess's `chessground` Flutter widget (GPL-3.0), so players get exactly the feel they're used to, for almost no work. It's named "Classic" with no Lichess name or logo, and its pieces and boards come from sets whose licenses allow redistribution.
  - **Doppel skin:** our own board widget, tuned by feel. This is where the work goes.
  - A skin can swap the widget and theme only. Game logic never forks per skin, so a fix lands in both.
- **Two isolated game surfaces:**
  - `features/doppel_game` talks to the Doppel backend and may show tilt indicators, pause, takebacks and move probabilities.
  - `features/lichess_game` talks to the Lichess Board API directly (OAuth2 PKCE, token on device).
- **The Lichess hard boundary, enforced in code:**
  - `lichess_game` may not import the backend or inference modules. An import-lint rule fails CI if it does.
  - The backend has no endpoint that accepts a live Lichess game id.
  - Post-game analysis unlocks only after Lichess reports the game finished.
  - While a Lichess game is live the client refuses to start Doppel play (a doppel's reply to a copied position is "move recommendations from software" under the Lichess ToS).
  - Lichess rules we design to (checked 2026-11-05): the Board API forbids "engine assistance, or any kind of outside help" and allows only Rapid, Classical and Correspondence seeks (Blitz only for direct challenges, games vs the AI and bulk pairing). Doppels play only from BOT accounts (Bot API, engine play allowed), casual while testing, and must follow the sandbagging/aborting/boosting rules. A doppel modelled on a real player must not be presented as that player (Lichess forbids impersonation).
- The API client is generated from `contracts/openapi.json`, so a backend change that breaks the client fails at build time.

## lab

- Today's `doppel/` package moves to `lab/doppel_lab/`, and `scripts/` and `results/` move under `lab/`. Data stays under `data/`, gitignored.
- It adds `export_package.py`: checkpoint → ONNX + manifest, and runs the parity test.
- GPU work goes through the warden: `warden run ... -- uv run --project lab python -m doppel_lab.X --gpu-ok`.
- The lab may read Lichess CC0 dumps and the consented game export. It never talks to the production database.

## Licensing (must decide before the first public commit)

| Component | Constraint | Consequence |
|---|---|---|
| Maia-3 | AGPL-3.0 | Doppel's residual on Maia-3 is a derivative, so serving it over a network obliges publishing the backend source. Fine for an open project. |
| dartchess | GPL-3.0 | The client must be GPL-compatible. Lichess ships GPL apps on both app stores. |
| ChessMimic | PolyForm Noncommercial | Use it only as a lab baseline. Never put it in the backend, the client or a release. |

**Decided 2026-11-04:** the whole repo is AGPL-3.0, the same as Lichess and Maia-3.

## Build order (client first)

1. Restructure the repo into this layout and move `doppel/` to `lab/`: about 30 min.
2. Build the Flutter client shell with the `BoardSurface` interface and the Classic skin (chessground), then the Doppel skin prototype (drag, tap-move, animation, haptics, sound, premove), playing a local random-move opponent, as an Android debug build plus a web build: 2–3 days, iterated by feel.
3. Point the client at today's `demo/server.py` for real Maia-3 moves: about 2 h.
4. Build `doppel_model`, the Maia-3 ONNX export, `backend/infer` and the `api` skeleton, replacing `demo/server.py`: about 1.5 days. **Done 2026-11-04:** `api` skeleton (games, clocks, per-game z, REST + WebSocket, SQLite; no accounts yet), and the client plays hosted games over it with server clocks and resume. All three Maia-3 sizes export with parity ≤ 3.6e-5 on 1,000 positions; infer serves 79M in about 21 ms per move on 4 ORT threads.
5. Accounts and consent, ratings, the consented export: **done 2026-11-05** (anonymous bearer-token accounts, versioned withdrawable training consent, MAP Bradley-Terry ratings, salted export with tombstones; see backend/api/README.md). Next: Lichess OAuth with the Board API, `lichess_bot`. **Done 2026-11-06:** Lichess sign-in (PKCE on device) and Board API play in the client; `lichess_bot` running as doppelbot behind an allowlist.

## Open decisions

- The cloud provider and region, which matters for latency to the first players.
- Whether doppels ever run on-device. A small package could, for offline play, and the ONNX format keeps that option open.

## Doppel skin: design direction (2026-11-04)

The target feel is a gamified board in the spirit of Duolingo's chess: bright colors, cartoonish chunky pieces, and snappy feedback (squash-and-stretch on pickup and drop, haptics on every move, punchy capture sounds, celebratory checkmates). The Classic skin exists for players who want the familiar look.

- **Inspiration, not copying.** We take the genre conventions: a saturated palette, rounded shapes, juicy feedback, and a material tracker. We copy none of Duolingo's assets (piece art, characters, sounds, animations), we don't imitate its trade dress closely enough to be mistaken for it, and we never use its name or brand in the app. All art and sound is original or openly licensed.
- **Material points tracker in every mode, including Lichess games.** It's a count of piece values on the board (P1 N3 B3 R5 Q9), not engine analysis, and Lichess's own UI shows the same material difference. It's computed locally from the position with no model call, so it stays inside the Lichess hard boundary. The skin can style it (a bar, a badge, an animated +N), but it may only ever show material.
