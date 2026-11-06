"""Dumb game analyzer: break one finished game into per-ply trajectories, derive cheap features, render HTML.

    python -m doppel_lab.analyzer <game> [--out game.html] [--no-maia] [--no-stockfish] [--nodes 50000]

<game> is a Lichess game ID or URL (finished games only, fetched from the public export API), a PGN file, or a
game_id from data/v0/games_*.parquet. Everything runs on the CPU in seconds for one game.

Layers, cheapest first (docs/SCIENCE.md §3 and §3.4):
  source.py      load a game into one record (moves, clocks, ratings, result)
  doppel_lab.features.channels    raw per-ply signals by compute tier (shared with corpus-scale baselines)
  doppel_lab.features.transforms  rolling mean, EWMA, volatility, velocity, drawdown; events; per-game summary
  render.py      one self-contained HTML page: stacked panels on a shared ply axis, board on hover
"""
