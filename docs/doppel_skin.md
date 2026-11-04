# Doppel skin spec

The aim is a board that feels like a game: dark, bold and reactive. These are the user's notes on what makes Duolingo's chess board sticky (2026-11-04), turned into requirements. We take the ideas and build all our own art, sound and motion. Nothing here copies Duolingo's assets or exact colours.

## Status (2026-11-04)

v1 items 1–5 are shipped and played on the Pixel 8. The user rated the feel "good enough", so the game-feel workstream is **shelved**. Resume with item 8 (own board widget, squash-and-stretch) or item 6 (avatars).

## v1 (first pass)

| # | Requirement | Notes |
|---|---|---|
| 1 | **Dark-mode screen; dark squares equal the app background** | Light squares are a lighter shade of the same dark slate, so the board reads as part of the screen rather than a separate panel. |
| 2 | **Minimal, smooth pieces with very thick bold outlines** | An original SVG set: simple curved silhouettes and a heavy outline, sticker style. White pieces are cream with a near-black outline; black pieces are charcoal with a light outline so they stay visible on dark squares. |
| 3 | **Material balance badge, green when ahead and red when behind** | The standard values: P1, N3, B3, R5, Q9. Computed from the position only, so it's allowed in Lichess games too. It animates when the value changes. |
| 4 | **Snappy feedback** | Short move animation (about 120 ms), haptics on every move (a stronger one for captures and checks), and original synthesized sounds for move, capture and check. |
| 5 | **Lightning strike when a queen is captured** | A flash plus a jagged bolt onto the capture square and a screen shake. It fires for either side's queen. |

## Later (directional)

| # | Requirement | Notes |
|---|---|---|
| 6 | **Opponent avatar with facial expressions** | Thoughtful, with eyes scanning the board, after my move. Worried when I capture or give check. Smug when they capture or check me. Each doppel gets its own avatar, which fits persistent identities. Likely built as a Rive state machine driven by game events. |
| 7 | **Opponent reacts to big moments** | The queen-capture lightning plus an avatar reaction. Later, reactions could reflect the doppel's actual tilt z: a tilted doppel looks rattled. That is a feature only Doppel can offer. |
| 8 | **Own board widget** | Squash-and-stretch on pickup and drop needs our own renderer. v1 reuses chessground with our own palette and pieces behind the same `BoardSurface` interface. |

## Event hooks

Skins derive these events from consecutive `BoardView`s, so game logic stays shared:

- `move`
- `capture(role)`
- `check`
- `checkmate`
- `queenCaptured(by side)`
- `materialSwing(delta)`

Avatars (#6) consume the same events.
