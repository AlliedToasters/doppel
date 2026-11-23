#!/usr/bin/env bash
# After the kpair chain: keep/knock-out interventions on neuron sets (neuron_attrib causal), one warden job per
# band (resumable per band). Rerun after a reboot.
cd /home/alliedtoasters/projects/chess/bot
W=~/projects/clearwood/warden/.venv/bin/warden
while pgrep -f "attrib_chain.sh|kpair_chain.sh" > /dev/null; do sleep 60; done
for b in 700 1100 1500 1900 2300; do
  echo "$b: $($W run --vram 16000 --max-runtime 5400 --label doppel:neuron_causal_$b --cwd $PWD --wait -- \
        uv run --project lab python -m doppel_lab.neuron_attrib causal --bands $b --gpu-ok 2>&1 | tail -1)"
done
echo CAUSAL DONE
