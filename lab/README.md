# lab

Research sandbox, runs on the dev box. Run everything from the repo root (paths like `data/...` are root-relative):

    uv run python -m doppel_lab.<module> ...          # CPU
    warden run --vram <MiB> --max-runtime <s> --label doppel:<what> --cwd $PWD -- \
      uv run --project $PWD python -m doppel_lab.<module> ... --gpu-ok   # GPU, queued
    PYTHONPATH=. .venv/bin/python lab/scripts/<script>.py

Data lives in ../data (gitignored); results in lab/results.

## Canonical features

Per-move feature tables for baselines, from the same code the analyzer uses (docs/SCIENCE.md §3):

    uv run python -m doppel_lab.features.table --split val --tier T2    # T0 moves/clocks, T1 Maia-3, T2 Stockfish

`doppel_lab.features.registry.select(tier="T1", regime="R1")` lists the features a baseline at that tier may use.

## Position book (Rust)

`lab/poskey` counts Polyglot Zobrist position keys (equal to `chess.polyglot.zobrist_hash`) through a Bloom
filter in front of an exact map. One month of train games takes about a minute on 26 threads:

    cargo build --release --manifest-path lab/poskey/Cargo.toml
    uv run python -m doppel_lab.posbook probe && uv run python -m doppel_lab.posbook run full
    lab/poskey/target/release/poskey count --out DIR [--fpr 0.001] [--band-min 1500 --band-max 1500] FILES...
    lab/poskey/target/release/poskey count --key prefix ...         # book.py's move-order key (R1) instead
    lab/poskey/target/release/poskey annotate --out DIR FILES...    # per-ply keys of every game, in order
    zstd -dc db.pgn.zst | lab/poskey/target/release/poskey pgn --out DIR [--lichess-blitz]   # PGN -> parquet

The explorations built on it (`doppel_lab/posbook_{behave,transpose,personal,otb,months,charts}.py`) write
`lab/results/posbook_explore.json` (docs/SCIENCE.md §8, Book depth).

## Render any finished game

    uv run python -m doppel_lab.analyzer <lichess game ID or URL | file.pgn> [--out game.html] [--tables DIR]

Writes one self-contained HTML page with every per-ply trajectory (clocks, think time, material, Maia-3, Stockfish)
on a shared axis, the board at the hovered ply, and a per-side summary (docs/SCIENCE.md §3.4). Finished games
only. Needs Stockfish on the PATH (or `--stockfish <path>`, or `--no-stockfish`); the first run downloads the
Maia-3 79M weights from Hugging Face (`--no-maia` skips it). CPU only, about 10 s per game.
