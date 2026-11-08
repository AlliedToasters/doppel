#!/usr/bin/env bash
# Ship the committed tree to the VM and (re)start the stack. Builds the landing page here; copies Maia-3 model
# packages only (ChessMimic is never deployed). First run creates deploy/.env on the VM.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/.config/doppel/deploy.env
H=doppel@$DOPPEL_HOST
[ -z "$(git status --porcelain -- backend packages deploy site pyproject.toml uv.lock)" ] || { echo "commit first: shipping HEAD" >&2; exit 1; }
(cd site && npm ci --silent && npm run build >/dev/null)
ssh "$H" 'mkdir -p ~/doppel/models ~/doppel/site'
git archive HEAD pyproject.toml uv.lock packages backend lab/pyproject.toml deploy .dockerignore | ssh "$H" 'tar -x -C ~/doppel'
rsync -a --delete site/dist/ "$H:doppel/site/dist/"
rsync -a --delete --include='maia3-*/***' --exclude='*' models/ "$H:doppel/models/"
if ! ssh "$H" test -f doppel/deploy/.env; then
  printf 'DOMAIN=doppelbot.org\nPOSTGRES_PASSWORD=%s\nCF_API_TOKEN=%s\nINFER_THREADS=4\n' \
    "$(openssl rand -hex 24)" "$(cat ~/.config/doppel/cloudflare.token)" | ssh "$H" 'umask 077; cat > doppel/deploy/.env'
fi
ssh "$H" 'cd doppel/deploy && docker compose up -d --build --remove-orphans 2>&1 | tail -3 && docker compose ps --format "{{.Service}} {{.Status}}"'
