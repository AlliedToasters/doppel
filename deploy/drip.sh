#!/usr/bin/env bash
# Daily blog drip: publish the top post of the approved queue (site/ /queue/), commit, push and ship.
# Does nothing when the queue is empty. Refuses to run with uncommitted changes under the shipped paths, so a
# half-edited post is never swept into a release. Run by the user timer doppel-drip.timer; log:
# ~/.local/state/doppel/drip.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.nvm/versions/node/v24.13.0/bin:$PATH"
echo "== $(date -Is)"
if [ -n "$(git status --porcelain -- backend packages deploy site client pyproject.toml uv.lock)" ]; then
  echo "skip: uncommitted changes under shipped paths"; git status --short -- site backend packages deploy client; exit 1
fi
cd site
if ! node scripts/publish-next.mjs --dry >/dev/null 2>&1; then echo "queue empty: nothing to publish"; exit 0; fi
msg=$(node scripts/publish-next.mjs)
echo "$msg"
cd ..
git add site/src/content/blog
git commit -q -m "site: ${msg%%;*} (daily drip)"
git push -q origin main
deploy/ship.sh
echo "shipped"
