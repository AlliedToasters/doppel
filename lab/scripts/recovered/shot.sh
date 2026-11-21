#!/usr/bin/env bash
# usage: shot.sh <slug> [width]  -> scratchpad/shots/<slug>-<width>.png (full post, tall window)
HS=~/.cache/ms-playwright/chromium_headless_shell-1200/chrome-headless-shell-linux64/chrome-headless-shell
W=${2:-760}; D=/tmp/claude-1000/-home-alliedtoasters-projects-chess/85159de4-f4b9-413c-b0f9-3c6818a37987/scratchpad/shots
timeout 60 $HS --no-sandbox --hide-scrollbars --window-size=$W,${3:-6000} --virtual-time-budget=5000 \
  --screenshot=$D/$1-$W.png http://100.101.146.35:4321/blog/$1/ >/dev/null 2>&1; echo $D/$1-$W.png
