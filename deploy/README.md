# deploy

Stage 1: one cloud CPU VM (Hetzner) behind Cloudflare, running `caddy + api + infer + postgres` under docker compose,
with `lichess_bot` optional. Domain: doppelbot.org.

| Path | Served by |
|---|---|
| `/` | `site/` (Astro landing page), static |
| `/play/` | `client/` Flutter web build, static |
| `/api/*` | `backend/api` (REST + WebSocket); it calls `infer` on the internal network |

## Build the static parts (on bigbox)

    (cd site && npm ci && npm run build)                          # -> site/dist
    (cd client && flutter build web --base-href /play/ --dart-define=BACKEND_URL=https://doppelbot.org)

## On the VM

    cp .env.example .env                  # set POSTGRES_PASSWORD
    rsync models/ <vm>:doppel/models/     # Maia-3 packages only; ChessMimic is never deployed
    docker compose up -d --build

`DOPPEL_REQUIRE_AUTH=1`: every game needs an account token.

The Lichess bot is off by default because only one host may run the doppelbot account at a time. To move it,
stop `doppel-lichess-bot` on bigbox, put the token in `deploy/secrets/lichess_bot.token`, then run
`docker compose --profile bot up -d`.

## Cloudflare

Proxied A/AAAA records for `@` and `www` point at the VM. Set SSL mode to Full (strict). Caddy gets its own
Let's Encrypt certificate over HTTP-01; Cloudflare passes `/.well-known/acme-challenge/` through.

## Local smoke test

    DOMAIN=localhost docker compose -p doppeltest -f compose.yaml -f <override mapping 127.0.0.1:18443:443> up -d

Checked 2026-11-07: `/`, `/play/`, `/api/bots` and a 404 page all serve, `www` redirects, unauthenticated
games get 401, and a game with a token plays e2e4 e7e5 against Postgres.
