# GPU backlog

Standing rule: the card should never sit idle while this list has a ready item. Every item is real chess research.
Submit through the warden only (`warden run ... --label doppel:<what> -- uv run --project <repo> python -m
doppel_lab.<module> ... --gpu-ok`), one or two queued at a time, and write each result into docs/SCIENCE.md before
the item leaves this file. New training runs that change the shipped Doppel model still need the user's go-ahead;
everything here is lab research.

## Running / queued

| Job | What it answers | Approx. GPU time |
|---|---|---|
| `evalnet_ft train --games 100000 --epochs 3` | does 3× more training move blunder recall off 0.59? | 100 min |
| `evalnet_ft label --games data/v0/games_1500_train.parquet --out data/evals/v0_train_evalnet_ft.parquet` | T2-lite evals for all 700k v0 games (49M positions); unlocks the style study at scale | 70 min |

## Ready (code exists; just submit)

1. **Style traits at scale.** After the v0 label job: rerun `sharpness style` on the 700k v0 games with net evals.
   Today it covers 873 players; the v0 corpus has 21,659 players with 20+ games. Tests whether steering and time
   allocation stay stable, rating-free and unprofitable on 25× the players. (CPU after the label job.)
2. **Fixed-1500 Maia teacher over v0 val + a fresh month.** `teacher.py --games <parquet> --out <dir> --fixed-elo
   1500`: T1 features for any new corpus build (see item 5).

## Needs a little code first

3. **v0 data-scaling run (answers "would a bigger corpus help v0?").** The v0f recipe on 50% of the training
   games, everything else equal. Compare val ce_post − ce_teacher, z active units, z-swap gain and the players
   suite re-ID against v0f. If halving the data costs re-ID, doubling it will help. Needs a `--train-frac` flag in
   `v0_train.py`. About 3.75 h.
4. **Eval-net distillation sweep.** Fine-tune the 23M Maia-3 trunk (if available) on the same labels: is a 3×
   cheaper net good enough for screening? About 40 min.
5. **Full-month corpus (more games per player).** The v0 corpus is the head of the 2026-10 file: median player has
   3 games, only 8,621 players have 40+. A full month, filtered to the 1500 band, gives the z encoder many more
   context games per player. CPU: download + parse. GPU: the Maia-3 teacher pass (about 3–4 h per 50M positions)
   and net evals (about 70 min per 50M).
6. **Other bands.** The same v0 pipeline at 1100 and 1900, so the Doppel is not 1500-only. GPU: a teacher pass and a
   training run per band.

## Done

- 2026-11-09: evalnet labels for all 36k corpus games (3.5 min). Net T2+T1 on 36k matches Stockfish T2+T1 on 12k
  (214 vs 214). SCIENCE.md §3.1.
