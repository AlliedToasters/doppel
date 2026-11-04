# lab

Research sandbox, runs on the dev box. Run everything from the repo root (paths like `data/...` are root-relative):

    uv run python -m doppel_lab.<module> ...          # CPU
    warden run --vram <MiB> --max-runtime <s> --label doppel:<what> --cwd $PWD -- \
      uv run --project $PWD python -m doppel_lab.<module> ... --gpu-ok   # GPU, queued
    PYTHONPATH=. .venv/bin/python lab/scripts/<script>.py

Data lives in ../data (gitignored); results in lab/results.
