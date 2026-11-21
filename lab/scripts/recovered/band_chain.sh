#!/usr/bin/env bash
# Wait for a band's v0e_vec train job, build the context norm, then queue v0f training and its eval.
# usage: band_chain.sh <band> <vec_train_job_id>
set -euo pipefail
B=$1; JOB=$2
W=~/projects/clearwood/warden/.venv/bin/warden
REPO=/home/alliedtoasters/projects/chess/bot
cd "$REPO"
$W wait "$JOB" >/dev/null || true
st=$($W status "$JOB" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['state'],d.get('rc'))")
echo "vec $B: $st"
[ "$st" = "done 0" ] || { echo "vec job failed; stopping"; exit 1; }
(cd lab && uv run python -m doppel_lab.v0e_data norm --band "$B")
T=$($W run --vram 20000 --max-runtime 14400 --label doppel:v0f-$B-train --cwd "$REPO" -- \
    uv run --project "$REPO/lab" python -m doppel_lab.v0e_train train --run v0f_$B --band "$B" --w-con 0.1 --gpu-ok \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['job_id'])")
echo "train $B queued: $T"
$W wait "$T" >/dev/null || true
st=$($W status "$T" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['state'],d.get('rc'))")
echo "train $B: $st"
[ "$st" = "done 0" ] || { echo "train failed; stopping"; exit 1; }
E=$($W run --vram 12000 --max-runtime 3600 --label doppel:v0f-$B-eval --cwd "$REPO" -- \
    uv run --project "$REPO/lab" python -m doppel_lab.v0e_train eval --run v0f_$B --band "$B" --gpu-ok \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['job_id'])")
$W wait "$E" >/dev/null || true
echo "eval $B: $($W status "$E" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['state'],d.get('rc'))")"
