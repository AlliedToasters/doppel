# site

Landing page for doppelbot.org (Astro, static). `npm run dev` for local work; `npm run build` writes `dist/`,
which Caddy serves at `/` (see `deploy/`). Fonts are self-hosted (`@fontsource-variable/nunito`), so the page makes no
third-party requests. Pieces are copied from `client/assets/pieces/doppel`.

## Lab notes (blog)

Posts live in `src/content/blog/` as `.md` or `.mdx`, with frontmatter `title`, `description`, `date` and
`draft`. A post with `draft: true` shows in `npm run dev` (marked "draft") and is left out of `npm run build`, so
it never ships. Flip it to `false` to publish, then run `deploy/ship.sh`.

Charts: in an `.mdx` post, import `src/components/LineChart.astro` and pass data. Copy result JSON from
`lab/results/` into `src/results/` and import it, so the numbers in a post come from the same file as the lab notes.
The chart renders to SVG at build time, with a hover readout and a "Show the data" table. Colors go by `slot`
(1 blue, 2 orange, 3 aqua, 4 yellow; validated for colour-blind separation on the panel colour), so give one
entity the same slot in every chart. Up to four series per chart.
