# GPU backlog

Standing rule: the card should never sit idle while this list has a ready item. Every item is real chess research.
Submit through the warden only (`warden run ... --label doppel:<what> -- uv run --project <repo> python -m
doppel_lab.<module> ... --gpu-ok`), one or two queued at a time, and write each result into docs/SCIENCE.md before
the item leaves this file. New training runs that change the shipped Doppel model still need the user's go-ahead;
everything here is lab research.

## Running / queued

| Job | What it answers | Approx. GPU time |
|---|---|---|
| `v0e_train train --run v0f_half --w-con 0.1 --tau 0.1 --train-frac 0.5 --max-steps 32808` | v0f on half the games at equal steps: is the corpus the bottleneck? Then `v0e_train eval --run v0f_half` and the players suite vs v0f | 3.75 h |

## Ready (code exists; just submit)

1. **Style traits at scale.** After the v0 label job: rerun `sharpness style` on the 700k v0 games with net evals.
   Today it covers 873 players; the v0 corpus has 21,659 players with 20+ games. Tests whether steering and time
   allocation stay stable, rating-free and unprofitable on 25× the players. (CPU after the label job.)
2. **Fixed-1500 Maia teacher over v0 val + a fresh month.** `teacher.py --games <parquet> --out <dir> --fixed-elo
   1500`: T1 features for any new corpus build (see item 5).

## Needs a little code first

3. **v0 data-scaling, quarter point.** If v0f_half shows a cost: `--train-frac 0.25` for a third curve point.
4. **Eval-net distillation sweep.** Fine-tune the 23M Maia-3 trunk (if available) on the same labels: is a 3×
   cheaper net good enough for screening? About 40 min.
5. **Full-month corpus (more games per player).** The v0 corpus is the head of the 2026-10 file: median player has
   3 games, only 8,621 players have 40+. A full month, filtered to the 1500 band, gives the z encoder many more
   context games per player. CPU: download + parse. GPU: the Maia-3 teacher pass (about 3–4 h per 50M positions)
   and net evals (about 70 min per 50M).
6. **Other bands.** The same v0 pipeline at 1100 and 1900, so the Doppel is not 1500-only. GPU: a teacher pass and a
   training run per band.

## Done

- 2026-11-09: evalnet labels for all 700k v0 train games (49M positions, 70 min) ->
  `data/evals/v0_train_evalnet_ft.part*.parquet`.
- 2026-11-09: evalnet fine-tune on 100k games × 3 epochs (99 min). MAE 2.78 → 2.62 win%, blunder recall 0.59 →
  0.62, still short of 1,000-node Stockfish (0.65). Checkpoint `data/probe/evalnet_ft_g100k_e3.pt`. SCIENCE.md §3.1.
- 2026-11-09: evalnet labels for all 36k corpus games (3.5 min). Net T2+T1 on 36k matches Stockfish T2+T1 on 12k
  (214 vs 214). SCIENCE.md §3.1.
