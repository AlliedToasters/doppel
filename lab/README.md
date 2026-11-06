# lab

Research sandbox, runs on the dev box. Run everything from the repo root (paths like `data/...` are root-relative):

    uv run python -m doppel_lab.<module> ...          # CPU
    warden run --vram <MiB> --max-runtime <s> --label doppel:<what> --cwd $PWD -- \
      uv run --project $PWD python -m doppel_lab.<module> ... --gpu-ok   # GPU, queued
    PYTHONPATH=. .venv/bin/python lab/scripts/<script>.py

Data lives in ../data (gitignored); results in lab/results.

## Render any finished game

    uv run python -m doppel_lab.analyzer <lichess game ID or URL | file.pgn> [--out game.html] [--tables DIR]

Writes one self-contained HTML page with every per-ply trajectory (clocks, think time, material, Maia-3, Stockfish)
on a shared axis, the board at the hovered ply, and a per-side summary (docs/SCIENCE.md §3.4). Finished games
only. Needs Stockfish on the PATH (or `--stockfish <path>`, or `--no-stockfish`); the first run downloads the
Maia-3 79M weights from Hugging Face (`--no-maia` skips it). CPU only, about 10 s per game.
