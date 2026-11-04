#!/usr/bin/env bash
# Wait for each self-play file, then Stockfish-annotate it (one at a time, after the human run ends).
cd "$(dirname "$0")/../.."
while pgrep -f "doppel_lab.analyze data/human" >/dev/null; do sleep 30; done
remaining=12
while (( remaining > 0 )); do
  remaining=0
  for ENGINE in maia3 chessmimic; do
    for BAND in 1100 1300 1500 1700 1900 2100; do
      IN=data/synth/${ENGINE}_${BAND}.parquet; OUT=data/evals/${ENGINE}_${BAND}.parquet
      [[ -f $OUT.done ]] && continue
      if [[ -f $IN ]]; then
        .venv/bin/python -m doppel_lab.analyze "$IN" "$OUT" --nodes 20000 --workers 28 && touch "$OUT.done"
      else
        remaining=$((remaining + 1))
      fi
    done
  done
  (( remaining > 0 )) && sleep 60
done
echo "all synthetic games analysed"
