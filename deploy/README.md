# deploy

Stage 1: one cloud CPU VM (Hetzner) behind Cloudflare, running `caddy + api + infer + postgres` under docker compose,
with `lichess_bot` optional. Domain: doppelbot.org.

| Path | Served by |
|---|---|
| `/` | `site/` (Astro landing page), static |
| `/account/` | part of `site/`: sign in with Lichess, training consent, delete account |
| `/api/*` | `backend/api` (REST + WebSocket); it calls `infer` on the internal network |

## Ship

    deploy/provision.sh   # once: Hetzner VM (ccx23, fsn1) + firewall + Cloudflare DNS and TLS Full (strict)
    deploy/ship.sh        # every release: builds site/, copies HEAD + Maia-3 packages, docker compose up

Tokens live outside git in `~/.config/doppel/`: `hcloud.token` (Hetzner, Read & Write) and `cloudflare.token`
("Edit zone DNS" template on doppelbot.org; set SSL/TLS to Full (strict) by hand unless it also has Zone Settings edit). `provision.sh` writes `deploy.env` (the VM's IP).
Caddy gets certificates over DNS-01 with the Cloudflare token, so it works behind the proxy.

`/play/` (Flutter web) is not served: dartchess uses 64-bit integer literals, which dart2js rejects, and
`flutter build web --wasm` still builds the JS fallback.

The Lichess bot is off by default because only one host may run the doppelbot account at a time. To move it,
stop `doppel-lichess-bot` on bigbox, put the token in `deploy/secrets/lichess_bot.token`, then run
`docker compose --profile bot up -d`.

## Local smoke test

    DOMAIN=localhost PUBLIC_URL=https://localhost:18443 CF_API_TOKEN=<any 40 chars> docker compose -p doppeltest -f compose.yaml -f <override mapping 127.0.0.1:18443:443> up -d

Checked 2026-11-07: `/`, `/account/`, `/api/bots` and a 404 page all serve, `/account/` signs in from a login code, `www` redirects, unauthenticated
games get 401, and a game with a token plays e2e4 e7e5 against Postgres.
