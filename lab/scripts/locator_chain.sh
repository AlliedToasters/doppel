#!/usr/bin/env bash
# Locator cold-start test (doppel_lab.locator), one warden job per band, then ntrain 15. Resumable per band.
# Can be dropped into data/allie/grow/queue/ to run between restore segments.
cd /home/alliedtoasters/projects/chess/bot
W=~/projects/clearwood/warden/.venv/bin/warden
for n in ${NTRAINS:-5 15}; do
  for b in ${BANDS:-700 1100 1500 1900 2300}; do
    echo "n$n $b: $($W run --vram 20000 --max-runtime 5400 --label doppel:locator_n${n}_$b --cwd $PWD --wait -- \
          uv run --project lab python -m doppel_lab.locator run --bands $b --ntrain $n --gpu-ok 2>&1 | tail -1)"
  done
done
echo LOCATOR DONE
