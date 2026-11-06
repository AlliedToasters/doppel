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

**Ledger and flood gate.** Every challenge and its decision, and every finished game, goes to `data/lichess_bot/ledger.jsonl`.
- If there are more than 20 challenges a minute overall, or more than 5 in 10 minutes from one account, the runner stops answering challenges and lets Lichess expire them. That account is then ignored for an hour.
- Because it sends nothing back, a flood costs only reading the event stream, and Lichess cannot rate-limit the bot's token.
- While the gate is shedding, the ledger writes one summary line a minute instead of one line per challenge. The episode is logged as `flood gate: SHEDDING` or `recovered`.

    .venv/bin/python -m doppel_lichess_bot.ledger --days 7     # rates, decisions, top challengers, results, floods

**Hard circuit breaker.** More than 100 challenges in 10 minutes means a sustained flood, which shedding is not meant to absorb. The runner then:
- writes `data/lichess_bot/BREAKER_TRIPPED`;
- disconnects from the Lichess event stream, so no challenge reaches bigbox at all;
- stays off, even across restarts, until a human deletes that file. It checks every 30 s.

Games already running finish on their own streams. To resume: `rm data/lichess_bot/BREAKER_TRIPPED`. To take the bot off Lichess on purpose, create the file.

