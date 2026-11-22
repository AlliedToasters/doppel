#!/usr/bin/env bash
# After the atlas chain: Tier 2 neuron attribution (neuron_attrib attrib) for bands 700 then 1900 as warden jobs of
# <= 95 min that resume per chunk, then the CPU report. Rerun after a reboot.
cd /home/alliedtoasters/projects/chess/bot
W=~/projects/clearwood/warden/.venv/bin/warden
while pgrep -f "atlas_chain.sh" > /dev/null; do sleep 60; done
R="uv run --project lab python -m doppel_lab.neuron_attrib"
for b in 700 1900; do
  for i in 1 2 3 4; do
    out=$($W run --vram 20000 --max-runtime 6600 --label doppel:neuron_attrib_$b --cwd $PWD --wait -- \
          $R attrib --bands $b --max-minutes 95 --gpu-ok 2>&1)
    log=$(echo "$out" | tail -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["log"])' 2>/dev/null)
    echo "$b pass $i: $(echo "$out" | tail -1)"
    [ -n "$log" ] && grep -q "attrib done" "$log" && break
  done
done
$R report && echo ATTRIB DONE
