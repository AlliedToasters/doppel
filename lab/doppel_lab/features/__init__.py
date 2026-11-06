"""Canonical per-ply features (docs/SCIENCE.md §3), shared by the game analyzer and every baseline.

  registry.py    every feature name with its tier (T0-T3), regime (R0-R2), timing (pre/act/post); VERSION
  channels.py    raw per-ply signals for one game: T0 moves and clocks, T1 Maia-3, T2 Stockfish
  transforms.py  R1 causal transforms, events, per-game summary
  table.py       corpus-scale per-move tables from stored data (games, teacher shards, evals)

    python -m doppel_lab.features.table --split val --tier T2
"""
