#!/usr/bin/env bash
# After the attribution chain: per-pair transfer vs distance (adapter_space kpair), one warden job per band
# (resumable per band), then the CPU report (NN ceiling, pair Spearman, boundary test). Rerun after a reboot.
cd /home/alliedtoasters/projects/chess/bot
W=~/projects/clearwood/warden/.venv/bin/warden
while pgrep -f "attrib_chain.sh" > /dev/null; do sleep 60; done
R="uv run --project lab python -m doppel_lab.adapter_space"
for b in 700 1100 1500 1900 2300; do
  echo "$b: $($W run --vram 16000 --max-runtime 3600 --label doppel:kpair_$b --cwd $PWD --wait -- \
        $R kpair --bands $b --gpu-ok 2>&1 | tail -1)"
done
$R kpair_report && echo KPAIR DONE
