#!/usr/bin/env bash
# After the kswap chain: exact per-neuron modulation of every adapter (restore bank, then split bank), each as
# warden jobs of <= 95 min that resume per chunk, then the CPU report. Rerun after a reboot.
cd /home/alliedtoasters/projects/chess/bot
W=~/projects/clearwood/warden/.venv/bin/warden
while pgrep -f "grow_loop.sh|kswap_chain.sh" > /dev/null; do sleep 60; done
R="uv run --project lab python -m doppel_lab.neuron_atlas"
for tag in _ctx200_restore _ctx200_fit0_s200_split; do
  for i in 1 2 3 4; do
    out=$($W run --vram 16000 --max-runtime 6600 --label doppel:atlas_mod$tag --cwd $PWD --wait -- \
          $R mod --tag $tag --max-minutes 95 --gpu-ok 2>&1)
    log=$(echo "$out" | tail -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["log"])' 2>/dev/null)
    echo "$tag pass $i: $(echo "$out" | tail -1)"
    [ -n "$log" ] && grep -q "mod done" "$log" && break
  done
done
$R report && echo ATLAS DONE
