#!/usr/bin/env bash
# After the restore loop ends: probe the new restored chunks, then kswap one band per warden job (resumable:
# probe skips probed chunks, kswap skips finished bands). Rerun after a reboot.
cd /home/alliedtoasters/projects/chess/bot
W=~/projects/clearwood/warden/.venv/bin/warden
while pgrep -f grow_loop.sh > /dev/null; do sleep 60; done
R="uv run --project lab python -m doppel_lab.adapter_space"
$W run --vram 16000 --max-runtime 6000 --label doppel:probe_restore --cwd $PWD --wait -- $R probe --gpu-ok | tail -1
for b in 2300 700 1100 1900 1500; do
  $W run --vram 16000 --max-runtime 6000 --label doppel:kswap_$b --cwd $PWD --wait -- $R kswap --bands $b --draws 4 --gpu-ok | tail -1
done
echo CHAIN DONE
