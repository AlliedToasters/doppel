#!/usr/bin/env bash
# Fetch third-party code at the pinned commits. vendor/ is gitignored.
#   maia3      AGPL-3.0                 used by lab and (later) backend
#   chessmimic PolyForm Noncommercial   lab baseline ONLY; never ship it in backend/ or client/
set -euo pipefail
cd "$(dirname "$0")"
fetch(){ [ -d "$1" ] || git clone "$2" "$1"; git -C "$1" fetch -q && git -C "$1" checkout -q "$3"; }
fetch maia3 https://github.com/CSSLab/maia3.git 1e13597
fetch chessmimic https://github.com/thomasj02/1e4_ai.git 8fcca23
echo "chessmimic model weights use git-lfs: run 'git -C chessmimic lfs pull' if backend/models is empty"
