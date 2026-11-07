# site

Landing page for doppelbot.org (Astro, static). `npm run dev` for local work; `npm run build` writes `dist/`,
which Caddy serves at `/` (see `deploy/`). Fonts are self-hosted (`@fontsource-variable/nunito`), so the page makes no
third-party requests. Pieces are copied from `client/assets/pieces/doppel`.
