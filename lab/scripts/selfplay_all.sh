#!/usr/bin/env bash
# Usage: scripts/selfplay_all.sh <engine> <device> [games]   (engine: chessmimic | maia3)
set -euo pipefail
cd "$(dirname "$0")/../.."
ENGINE=$1; DEVICE=$2; GAMES=${3:-1000}
EXTRA=(); [[ $DEVICE == cuda* ]] && EXTRA+=(--gpu-ok)
for BAND in 1100 1300 1500 1700 1900 2100; do
  OUT=data/synth/${ENGINE}_${BAND}.parquet
  [[ -f $OUT ]] && { echo "skip $OUT"; continue; }
  .venv/bin/python -m doppel_lab.selfplay --engine "$ENGINE" --band "$BAND" --games "$GAMES" \
    --human data/human/games_2026-10.parquet --out "$OUT.tmp" --device "$DEVICE" "${EXTRA[@]}"
  mv "$OUT.tmp" "$OUT"
done
