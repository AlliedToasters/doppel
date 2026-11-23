#!/usr/bin/env bash
# Cold-start test (adapter_space coldstart): one warden job per band, serial. Resumable per band.
cd /home/alliedtoasters/projects/chess/bot
W=~/projects/clearwood/warden/.venv/bin/warden
for b in ${BANDS:-700 1100 1500 1900 2300}; do
  echo "$b: $($W run --vram 16000 --max-runtime 3600 --label doppel:coldstart_$b --cwd $PWD --wait -- \
        uv run --project lab python -m doppel_lab.adapter_space coldstart --bands $b --gpu-ok 2>&1 | tail -1)"
done
echo "COLD DONE"
