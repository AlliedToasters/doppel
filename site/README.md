# site

Landing page for doppelbot.org (Astro, static). `npm run dev` for local work; `npm run build` writes `dist/`,
which Caddy serves at `/` (see `deploy/`). Fonts are self-hosted (`@fontsource-variable/nunito`), so the page makes no
third-party requests. Pieces are copied from `client/assets/pieces/doppel`.

## Lab notes (blog)

Posts live in `src/content/blog/` as `.md` or `.mdx`, with frontmatter `title`, `description`, `date`, `status`
(`draft`, `approved` or `published`) and, for approved posts, `queue` (publishing order). Only `published` posts
are built; `npm run dev` shows the rest, badged.

**Post queue.** In `npm run dev`, open `/queue/`: edit a post with a live preview (Save or Ctrl+S re-renders it),
approve it into the queue, reorder the queue, or send it back to draft. The page and its API (`queue/`) exist in
the dev server only and never ship. Writes need an `X-Queue` header and a same-origin request.
To serve it on the tailnet: `npx astro dev --host <tailnet IP>`.

**Publishing** takes one post at a time from the top of the queue:
`node scripts/publish-next.mjs` (sets `status: published` and today's date), then commit, push and `deploy/ship.sh`.

Charts: in an `.mdx` post, import `src/components/LineChart.astro` and pass data. Copy result JSON from
`lab/results/` into `src/results/` and import it, so the numbers in a post come from the same file as the lab notes.
The chart renders to SVG at build time, with a hover readout and a "Show the data" table. Colors go by `slot`
(1 blue, 2 orange, 3 aqua, 4 yellow; validated for colour-blind separation on the panel colour), so give one
entity the same slot in every chart. Up to four series per chart.
