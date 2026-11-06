# backend/lichess_bot

One long-lived runner for the Lichess BOT account `doppelbot`. It reads the Bot API event stream, accepts challenges and plays each game through `backend/infer`.

- **Strength:** the doppel plays at the challenger's own rating for that speed, clamped to 600–2600.
- **Think time:** human-like, from `doppel_model.heuristic_think_ms`, the same heuristic the API uses.
- **Accepts:** standard chess with a real-time clock from 1 to 60 minutes, casual only (`--rated` changes that). It declines other bots, and anything beyond `--max-games` (default 2) running at once.
- **Allowlist:** `--allow a,b` accepts challenges only from those Lichess users. Not set in production since 2026-11-06, when the repo went public. The AGPL source link is posted in every game's chat.
- **Greeting:** it opens each game by telling the opponent it is a human-like model and that its think times are simulated.

Runs as the user unit `doppel-lichess-bot` (~/.config/systemd/user/doppel-lichess-bot.service; `systemctl --user restart doppel-lichess-bot`, logs in logs/). The command below is what the unit runs.

```
.venv/bin/python -m doppel_lichess_bot.runner --allow <lichess names> >> logs/lichess_bot.log 2>&1 &
```

The token lives in `~/.config/doppel/lichess_bot.token` (chmod 600, `bot:play` scope) and is never logged. Change the allowlist in the unit's ExecStart, then `systemctl --user daemon-reload && systemctl --user restart doppel-lichess-bot`.
