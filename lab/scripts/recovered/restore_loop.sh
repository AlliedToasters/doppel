#!/usr/bin/env bash
# Submit restore segments one at a time until the first $1 index units are done (or a segment fails).
cd /home/alliedtoasters/projects/chess/bot
W=~/projects/clearwood/warden/.venv/bin/warden
for seg in 1 2 3 4 5; do
  out=$($W run --vram 16000 --max-runtime 7200 --label doppel:ttt_restore_seg$seg --cwd $PWD --wait -- \
        uv run --project lab python -m doppel_lab.allie_ttt restore --upto $1 --max-minutes 100 --gpu-ok)
  echo "seg $seg: $(echo "$out" | tail -1)"
  log=$(echo "$out" | tail -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["log"])')
  tail -1 "$log"
  grep -q "restore: $1/$1 of the target done" "$log" && { echo ALL DONE; exit 0; }
  echo "$out" | tail -1 | grep -q '"rc": 0' || { echo "segment failed"; tail -20 "$log"; exit 1; }
done
