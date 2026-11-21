#!/usr/bin/env bash
# Queue-friendly v0f training: segments of <= SEG minutes (each a self-contained warden job that resumes from
# ckpt.pt), one Doppel entry submitted at a time, then the eval. usage: train_segments.sh <band>
set -euo pipefail
B=$1; SEG=${SEG:-100}
W=~/projects/clearwood/warden/.venv/bin/warden
REPO=/home/alliedtoasters/projects/chess/bot
RUN=$REPO/data/v0/runs/v0f_$B
cd "$REPO"
state() { $W status "$1" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['state'],d.get('rc'))"; }
jid() { python3 -c "import json,sys;print(json.load(sys.stdin)['job_id'])"; }
for seg in $(seq 1 6); do
  J=$($W run --vram 20000 --max-runtime $(( (SEG + 15) * 60 )) --label doppel:v0f-$B-train-seg$seg --cwd "$REPO" -- \
      uv run --project "$REPO/lab" python -m doppel_lab.v0e_train train --run v0f_$B --band "$B" --w-con 0.1 \
      --max-minutes "$SEG" --gpu-ok | jid)
  echo "$(date +%H:%M) segment $seg queued: $J"
  $W wait "$J" >/dev/null || true
  st=$(state "$J"); echo "$(date +%H:%M) segment $seg: $st"
  [ "$st" = "done 0" ] || { echo "segment failed; stopping"; exit 1; }
  grep -q "training done" "$HOME/.local/state/warden/jobs/$J.log" && break
done
E=$($W run --vram 12000 --max-runtime 3600 --label doppel:v0f-$B-eval --cwd "$REPO" -- \
    uv run --project "$REPO/lab" python -m doppel_lab.v0e_train eval --run v0f_$B --band "$B" --gpu-ok | jid)
$W wait "$E" >/dev/null || true
echo "$(date +%H:%M) eval $B: $(state "$E")"
