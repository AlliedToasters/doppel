#!/usr/bin/env bash
# Grow the restored ctx200 adapter bank to the full index, one warden job at a time. Side jobs dropped in
# $S/queue/*.sh run between segments. Each segment resumes from its saved parts, so after a reboot just rerun:
#   nohup lab/scripts/grow_loop.sh [seg] > data/allie/grow/loop.log 2>&1 &
cd /home/alliedtoasters/projects/chess/bot
S=/home/alliedtoasters/projects/chess/bot/data/allie/grow   # queue/, queue_done/, STOP: survive reboots
mkdir -p $S/queue $S/queue_done
W=~/projects/clearwood/warden/.venv/bin/warden
T=185
seg=${1:-7}
while true; do
  for q in $(ls $S/queue/*.sh 2>/dev/null | sort); do
    echo "side job $(basename $q) start $(date +%H:%M)"
    bash $q > $S/queue_done/$(basename $q .sh).out 2>&1; echo "side job $(basename $q) rc=$? $(date +%H:%M)"
    mv $q $S/queue_done/
  done
  [ -f $S/STOP ] && { echo "STOP file: exiting"; exit 0; }
  out=$($W run --vram 16000 --max-runtime 7200 --label doppel:ttt_restore_seg$seg --cwd $PWD --wait -- \
        uv run --project lab python -m doppel_lab.allie_ttt restore --upto $T --max-minutes 100 --gpu-ok 2>&1)
  echo "seg $seg $(date +%H:%M): $(echo "$out" | tail -1)"
  log=$(echo "$out" | tail -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["log"])' 2>/dev/null)
  [ -n "$log" ] && tail -1 "$log"
  [ -n "$log" ] && grep -q "restore: $T/$T of the target done" "$log" && { echo ALL DONE; exit 0; }
  echo "$out" | tail -1 | grep -q '"rc": 0' || { echo "segment failed"; echo "$out" | tail -5; [ -n "$log" ] && tail -20 "$log"; exit 1; }
  seg=$((seg+1))
done
