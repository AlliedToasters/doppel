#!/usr/bin/env bash
# Block-5-only adapters (allie_ttt run --blocks 5) on 400 sampled sides per band, 200-game context.
# Compare with the 4-block ctx200 run on the same positions (doppel_lab.blocks_report).
cd /home/alliedtoasters/projects/chess/bot
W=~/projects/clearwood/warden/.venv/bin/warden
for b in ${BANDS:-700 1100 1500 1900 2300}; do
  echo "$b: $($W run --vram 16000 --max-runtime 5400 --label doppel:ttt_b${BLK:-5}_$b --cwd $PWD --wait -- \
        uv run --project lab python -m doppel_lab.allie_ttt run --band $b --ctx 200 --sample 400 --blocks ${BLK:-5} --gpu-ok 2>&1 | tail -1)"
done
echo "B${BLK:-5} DONE"
