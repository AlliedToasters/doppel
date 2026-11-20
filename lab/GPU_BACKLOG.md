# GPU backlog

Standing rule: the card should never sit idle while this list has a ready item. Every item is real chess research.
Submit through the warden only (`warden run ... --label doppel:<what> -- uv run --project <repo> python -m
doppel_lab.<module> ... --gpu-ok`). Tenants have equal priority: every job is self-contained and 1-2 h
(longer runs go in resumable segments, e.g. `v0e_train --max-minutes 100`), and at most one or two Doppel entries sit
in the queue at a time, and write each result into docs/SCIENCE.md before
the item leaves this file. New training runs that change the shipped Doppel model still need the user's go-ahead;
everything here is lab research.

## Running / queued

2026-11-19: adapter bank restore (unblocks 11, 11f, 12a, 12b). `allie_ttt restore --upto 25 --max-minutes 100`
(label `doppel:ttt_restore_seg<k>`, one segment at a time): rerun ctx200 chunks with their original seeds, saving the
adapters the full runs discarded. Order of record: `data/allie/restore_index_ctx200.json` (copy in
lab/results/), 185 units (band, chunk), bands in turn, chunks bit-reversal spread; first 5N units = N x 200 sides per
band; done = adapter file exists. Verified bit-exact vs the original run (1500 chunk 2000: hits, epochs, log-probs all
identical). Target now: first 25 units (1,000 sides/band), ~4.6 h. `allie_ttt verify --band <b>` per band after.

2026-11-13 evening: 59b0b0d91955 `doppel:v0g_1100-seg1` running, 912de0bbdcf7 `doppel:v0g_1900-seg1` queued (`v0e_train train --run v0g_<band> --band <band> --w-con 0.1 --kl-per-player --res-lr-mult 0.3 --max-minutes 90`; seg2 each after, then eval). Done: v0g_b seg1+2, allie_v0e eval of v0g_a and v0g_b on the 1400–1600 bucket (v0g_b + memory +0.26 over Maia-3). Earlier commands, for reference:

| Command | What it answers | Approx. GPU time |
|---|---|---|
| `v0e_train train --run v0g_a --w-con 0.1 --kl-per-player --max-minutes 15` (segment 3; segment 2 stopped at step 32,707 of 32,808, train KL 27 nats), then `v0e_train eval --run v0g_a --style` | v0g_a: does a per-player KL give z reach? (item 7) | ~5 min + ~25 min |
| `top1 --engine maia3 --human data/allie/test.parquet --bands -1 --games-per-band 100000 --skip-plies 10 --cut-clock 30 --device cuda --gpu-ok --save-positions data/allie/top1_positions_maia3-79m.parquet --out lab/results/top1_allie_protocol_maia3-79m.json` (`--vram 8000`) | Maia-3-79M on the exact Allie protocol (target 57.1%; item 9a) | ~15 min |

## Ready (code exists; just submit)

1. **Fixed-1500 teacher over a fresh month** once item 5's corpus exists (`--games <parquet> --fixed-elo 1500`).

## Needs a little code first

4. **Eval-net distillation sweep.** Fine-tune the 23M Maia-3 trunk (if available) on the same labels: is a 3×
   cheaper net good enough for screening? About 40 min.
5. **Full-month corpus (more games per player).** The v0 corpus is the head of the 2026-10 file: median player has
   3 games, only 8,621 players have 40+. A full month, filtered to the 1500 band, gives the z encoder many more
   context games per player. CPU: download + parse. GPU: the Maia-3 teacher pass (about 3–4 h per 50M positions)
   and net evals (about 70 min per 50M).
6. **Other bands.** The same v0 pipeline at 1100 and 1900, so the Doppel is not 1500-only. Done: game selection,
   Maia-3 teacher (train + val), net evals, resign labels, v0 index, valrich (600 held-out players each), v0e meta.
   v0f_1100 and v0f_1900 trained and written up. Left: v0e_suite per band (players suite is 1500-centred).

7. **v0g: give z reach (draft 2026-11-12; training needs the user's go-ahead).** Diagnosis from v0f's checkpoint
   and log: the encoder knows the player (re-ID 0.97 at k = 40), but the posterior barely leaves the prior. Per-dim
   variance of mu over players is ~0.01 against a prior SD of ~1, KL is 0.76–1.03 nats even at k = 20–40, and the
   decoder trains on z sampled with that noise, so it learns ~1 nat of style: one axis, which is why steering and
   conformity move together (ρ −0.82). Cause: the KL is charged per training sample (β 0.25 · KL / positions in
   the sample, K = 8), and a player with G train games is sampled ~G times an epoch, so their z pays the KL G
   times. Over training samples the median player has 31 games (q10 5, q90 95). An honest per-player ELBO charges
   it once. Change (one flag, `--kl-per-player`): weight each sample's KL by 1 / (its player's train games), with
   β and free bits as v0f. Arms, each the v0f recipe at 32.8k steps (≈ 2.9 h, so two resumable 90-min segments):
   - v0g_a: `--kl-per-player`
   - v0g_b: `--kl-per-player --res-lr-mult 0.3` (the residual also learns at 1/10 LR in v0f)
   Gate against v0f (1500 valrich, `v0e_train eval --style`): z-swap gain above 0.00115 with disjoint CIs; repeat
   fidelity share above 1.19%; style SD ratio above 3%; conformity z_adds CI covering or above 0; coupling |ρ| of the
   steering and conformity pushes below 0.8. Must not regress: top1_prior ≥ 0.5615, ce_prior − ce_teacher ≤
   +0.001, think r ≥ 0.70, re-ID k = 10 ≥ 0.67. Risk: z noise leaks into the prior-z policy; the marginal loss
   and the teacher-abort tripwire guard it. Code: ~40 min plus a CPU smoke.

8. **Future (not urgent): rework how z reaches the move policy.** Today z enters only the small residual
   (`v0_model.py` Residual: ctx Linear on the bare current board, ~2M params, additive Δ capped at ±10). The residual
   never sees Maia-3's logits or features, so it cannot express "sharpen or flatten Maia for this player"
   (conformity) without re-deriving Maia's preferences from the board; one suspect for z's backwards conformity
   and the steering/conformity coupling (hypothesis, unmeasured). Options, cheapest first:
   (a) z → per-player temperature on Maia's log-probs before Δ (checkable offline on the cached style passes);
   (b) feed Maia's pooled features or top-k logits into the residual;
   (c) FiLM/adapters inside the frozen trunk (gradient flows through frozen Maia to z; the trunk forward is
   `no_grad` today, so this changes it);
   (d) z next to Maia's rating embeddings, full fine-tune (the original brief's design).
   Trigger: when the cheap tricks (v0g KL, data) stop moving z-swap / repeat fidelity / conformity.

9. **Step-3 candidates (2026-11-12; not yet chosen, see SCIENCE.md §10 "Hypotheses" and "Lit review").**
   (a) Maia-3-79M top-1 / nats on the Allie test set under the exact 884,049 filter (short GPU job, or CPU if the
   per-position predictions were saved). (b) Noise-ceiling estimate from repeat visits and crowd agreement (CPU).
   (c) Player and history recovery for the Allie test set. IDs done 2026-11-12 (`allie_players.py`: 18,239 / 18,239
   games, 34,247 players, all sides named, data/allie/players.parquet). History running (`allie_history.py run`,
   14 months streamed one at a time at ~11 MB/s, ~12 h, resumable per month, log data/allie/history/run.log). From 2022-01 the same pass also writes data/allie/base/<month>.parquet: a rating-stratified sample of
   non-test players' rated blitz (BASE_RATE per 200-Elo bucket, hashed on game id), the base corpus matched to the
   test set's flat 800-2200 rating mix (2026-10 and v0 are not: 2026-10 q10/q90 1100/2044 vs test 908/2355).
   Memory ceiling on the test set itself (`allie_memory.py`, CPU): +0.26 pts over a population mode with history
   through 2022-05; rerun when all months land, then join to 9a's saved Maia-3 positions for the gain over Maia-3. (d) H1: self-play from Maia-3 at T = 1 / 0.87 / rating 1700, scored by the game-level discriminator
   (GPU for self-play, ~1 h). Also H4: several self-play games per synthetic player from shared openings,
   scored on repeat-consistency (human 77%) and game-share of think time / error size (human 9% / 3.4%). (e) H2: whole vs shuffled games through the same discriminator (CPU after d).
   (f) H3: BC-anchored adversarial / statistic-matching fine-tune (needs code; after v0g and d).

10. **Career dynamics: rating drift and stale history (backlogged 2026-11-17).** On Allie 1500, adapter and z +
   lookup gains fade with history age (span < 7 days +0.99, 90+ days ≈ 0; SCIENCE.md "Rating drift and staleness").
   Span is confounded with play frequency. Next tests: (a) recency weighting within a player (adapter sample weights by
   game age, lookup weights by age), (b) the same heavy player scored with a recent vs an old 40-game window,
   (c) players whose Elo crosses a band edge: per-band z / lookup heads break there, so a slow career state
   (career z + per-game tilt) is the design target.
11. **Player modality: how many playstyles, and does the count change with rating? (backlogged 2026-11-17; research
   order step 2 lit review done, step 3 not decided).** Representations: per-player adapter deltas (runs must save
   them), the Doppel z, interpretable departure-from-Maia profiles. Two definitions of "number of styles":
   predictive (k shared cluster adapters recovering most of the per-player gain) and geometric (within-band
   effective dimension, cluster stability, split-half reliability to remove 40-game noise). Compare across the 5
   buckets with rating controlled. Product payoff: a cluster adapter as cold start for players under 20 games (the
   adapter gains nothing measurable below that). Prior art: McIlroy-Young et al. NeurIPS 2021 (stylometry, 98% ID
   from 100 games, qualitative clustering), Omi et al. arXiv 2502.14998 (PEFT style vectors, 47,864 players, no
   cluster count), Carlson arXiv 2606.25176 (Elo-disentangled z residual on Maia-3, silhouette 0.06, rating R² 0.06,
   no by-rating diversity). Nobody reports a predictive style count or how it varies with rating. Aggregate results
   only; no other users' usernames in committed results. Ethics: opponent-profiling during a live game is outside
   Lichess fair play; do not build that.
   - **Measurement notes (2026-11-18).** (1) Normalize every per-band metric for headroom: gain as a share of the
     positions Maia-3 misses (ctx200 own gain 2.58 / 2.59 / 2.33 / 2.19% at 700 / 1100 / 1500 / 1900, vs raw
     +1.28 / +1.18 / +1.01 / +0.93). (2) The cross-player swap control is the cluster test: score each side with
     its own adapter, a same-cluster neighbour's, and a random other-cluster player's. Real clusters mean
     same-cluster swaps cost much less than the random swap (today about -2.4 to -2.7 pts at every band); the
     same-cluster cost as a share of the random cost is a per-band, headroom-free cluster-validity number.
     Conformity at the modality level = k falls with rating while the random-swap cost stays flat.
   - **11e. Prep study (scheduled games and consenting players only).** The predictive version yields a
     per-opponent mixture profile ("88% A, 6% B, ...") in k forward passes over the opponent's recent games, well
     under 1 s. If style is a continuum, the card is coordinates, not percentages. Primary outcome needs no matches:
     after prep, the subject predicts the opponent's moves on held-out positions (top-1). Arms: none, the 10 s card,
     30 min traditional game review, sparring vs the opponent's doppel; compare gain per minute of prep. Secondary:
     scored results in scheduled matches (noisy). Only worth running if the predictive k is small.
   - **11f. Interpretability of the style space.** Receipts that the representation is personal: +0.82 top-1 at 40
     games (all buckets), own vs another player's adapter at 700 +1.03 vs -1.88, split-half rho +0.89 (player_markov)
     and +0.49/+0.51 (style traits). Steps: (1) PCA of adapter deltas / z per band; (2) per axis, the positions where
     moving along it shifts the move distribution most, shown as boards with gaining and losing moves; (3) correlate
     axes with style traits, evalnet blunder rate, opening family, capture/check/pawn-push rates, risk appetite;
     (4) blind human test: coaches name or sort the two ends of an axis, inter-rater agreement; (5) count axes reliable
     across split halves, per band. Check every axis with and without opening repertoire, which can pose as style.
12. **Persona lessons from LLM interpretability (backlogged 2026-11-18; lit review done, decide pending).** Claim
    under test: a sequence model trained on many humans gains by modelling who is speaking, and RL-style training
    collapses that to one persona. Chess map: Maia / Allie / chess-GPT = base model; Elo input = system prompt;
    per-side adapter = one-PEFT-per-user (OPPU, Tan et al. EMNLP 2024); Leela / "Chessformer" (Monroe & Chalmers,
    arXiv 2409.12272) and AlphaZero = RL-only, one engine persona. Prior chess proof of the base-model half: Karvonen
    (COLM 2024, arXiv 2403.15498), a GPT on PGNs linearly encodes player Elo and a skill vector raises win rate up to
    2.6x. Maia-3 as we use it sees one position + Elo, so it cannot infer the player in context; the adapter does it
    in weights. Allie sees the game prefix and could. Imports, cheapest first:
   - **12a. Engine axis.** Assistant Axis analogue (Lu et al., Anthropic, arXiv 2601.10387): is PC1 of per-player
     adapter deltas an Elo / engine-likeness axis? If so, count styles (item 11) after projecting it out. Blocker:
     allie_ttt discards the best LoRA snapshot (only log-probs saved); save it per chunk (rank 8, blocks 5-8, small)
     before any rerun, or rerun one band.
   - **12b. Player vectors.** Persona vectors (Chen et al., arXiv 2507.21509): difference-of-means direction per
     player vs the adapter delta; the cross-player swap is the specificity check.
   - **12c. Does the base model model the player?** TalkTuner (Chen et al., arXiv 2406.07882) recovers user age,
     gender, education from chat activations. Probe Maia-3 activations for player identity within one Elo bucket;
     Carlson's silhouette 0.06 predicts weak, which would itself say the base model barely models the player.
   - **12d. Two kinds of diversity loss.** RLHF cuts per-input and across-input diversity (Kirk et al., ICLR 2024).
     Chess: per-position move entropy per bucket, and the random-swap cost. Today the swap cost is flat (-2.41 /
     -2.52 / -2.70 / -2.36 at 700-1900) while own gain shrinks: players do not become interchangeable with rating.
     Both numbers come from existing outputs (CPU).
   - **12e. Repertoire vs style.** Emergent misalignment (Betley et al. 2025; persona features, Wang et al. OpenAI,
     arXiv 2506.19823): narrow tuning moves a broad persona. Our adapters do the reverse: ttt40 (off after ply 40)
     helps, so the gain may be opening repertoire, not style. Split gain by ply and by in-book vs out-of-book; feeds 11f.
   - **12f. Is the player's share of a move's uncertainty set by how forcing the position is?** Chess is closer to LLMs on
     verifiable tasks (math, code) than to open text: most move variance is graded error, not expression. So the
     prediction is that adapter gain concentrates where several moves are near-equal and vanishes in forcing positions.
     Identity can still be strong (it accumulates over moves); this is about prediction gain only. Tiers: (1) CPU now:
     bin positions by Maia-3's own policy margin (p1 - p2) and entropy from the saved log-probs, gain per bin, headroom-
     normalised; (2) engine version: best-vs-second-best eval gap. evalnet.py scores positions, not moves, so this needs
     a pass over every legal child (~30x per position, GPU) or Stockfish multipv on a test subsample. Do (1) first; the
     engine tier only if Maia's margin and the engine gap might disagree in a way that matters.
   - **Result 2026-11-19 (12d, 12e, 12f tier 1; CPU, `persona_pilots`, lab/results/persona_pilots.json).** Join of
     the ctx200 hits with Maia-3's full policy is exact (argmax = Maia hit on 100% of positions, all bands).
     12f: the personal signal (own - control, pts) falls with Maia's margin p1 - p2 in every band, from +6.3 to +7.2
     in the least forcing quintile (+4.5 at 2300) to +0.1 to +0.5 in the most forcing; own gain is negative where
     Maia is confident (about -1 to -2% of misses). Margin-gated adapter (adapter only where margin < tau, tau tuned
     on one half of test games, scored on the other): 700 +1.31 vs +1.28 ungated, 1100 +1.30 vs +1.18, 1500 +1.18
     vs +1.01, 1900 +1.09 vs +0.93, 2300 +0.36 vs +0.04 (tau ~0.2-0.3). Product candidate; changes the shipped
     model, needs the user's go. 12d: Maia entropy falls with rating (1.56 / 1.39 / 1.32 / 1.25 / 1.13 nats)
     while the headroom-normalised swap cost stays flat (-4.9 / -5.5 / -6.2 / -5.6 / -5.1% of misses): per-input
     diversity shrinks, across-player diversity does not. 12e: normalised gain is front-loaded (ply 10-19: +4.9 to
     +8.9% of misses; ply 60+: -3.0 to +1.2) but own - control stays positive late (+0.8 to +2.1 pts at ply 60+), so
     not all repertoire; exact-repeat positions (memory `hist`) are rare (125-7,413 per band) and gain 7-13 pts.
   - **Adapter geometry, first look (2026-11-19).** Weight space is the wrong place: each side's LoRA A starts
     from its own random init, so dW = BA of two sides are near-orthogonal (cosine 0.0005, participation ratio =
     n). Function space instead (`adapter_space probe`: change in log-probs on 2,560 fixed probe positions, Fisher-
     weighted), then `adapter_space fs` (geometry) and `adapter_space swap` (GPU: nearest-neighbour vs random swap,
     cluster-soup adapters for k = 1..64, the predictive count).
   - **Function space (2026-11-19, `adapter_space probe/fs/traits`, lab/results/adapter_fs*.json, adapter_traits.json;
     2,257 restored adapters, 288-506 per band).** Effective dimension (participation ratio) rises with rating:
     47 / 67 / 78 / 86 / 99 (own-band probe positions), but noise inflates it too (split-half pending). k-means
     silhouettes 0.03-0.07 and falling with k: no discrete styles, a continuum (as Carlson's 0.06). 12a: PC1 (PC2 at
     1100/1500) is a conformity axis, the adapter's shift of probability onto Maia-3's top move (Spearman +0.69 to
     +0.90), with chosen epoch / norm; not a rating axis (traits vs Elo within band |rho| < 0.19). The second axis is
     pawn moves vs piece retreats / castling. The mean adapter moves mass off Maia's top move (-2.9 pts at 700,
     -1.35 at 2300) and trait spreads shrink with rating (top-1 shift sd 3.7 -> 1.8 pts).
   - **Split-half (2026-11-19, `allie_ttt run --ctx 200 --sample 200 --split`, ~13 min/band).** Two adapters per
     side on disjoint halves of its train games. On the test game the sister-half adapter keeps the full gain while
     another player's costs 1.3-2.8 pts (700 +1.04 vs -1.74, 1100 +1.73 vs -2.07, 1500 +0.50 vs -2.75, 1900
     +0.93 vs -2.31, 2300 +0.36 vs -1.35): the personal signal reproduces across disjoint games. Function-space
     reliability (`adapter_space split`) after the split probe.
   - **Predictive count: swaps and cluster soups (2026-11-19, `adapter_space swap`, lab/results/adapter_swap.json,
     ~50 min GPU).** Own adapter reproduces the original hits (agreement 0.9999-1.0). Scoring each side with the
     r-th most similar other side (function-space cosine) gives a gradient: nn1 -1.4 to -2.5 pts, random -2.3 to
     -3.4, least similar -3.8 to -5.8; similarity is real but even the nearest neighbour's adapter is worse than
     Maia-3 (nn1 cost as a share of the random cost: 0.65 / 0.81 / 0.65 / 0.83 / 0.94). Cluster soups (mean dW of a
     k-means cluster, cross-fit so a side's own adapter is never in its soup, scale chosen on the other half):
     k = 1-32 recover -0.5 to +0.6 pts (best: 700 k16 +0.58 of own +1.75; 1900 k32 +0.29 of +1.49; 2300 none).
     Verdict: no small k recovers the per-player gain at any rating; the predictive "number of styles" is large.
     Cold start from a cluster adapter is weak (at most a third of the gain, 700 only). 11e's "88% A, 6% B" card
     has no basis; a trait card does (below).
   - **Split-half reliability in function space (2026-11-19, `adapter_space split`, adapter_split.json,
     adapter_traits_splithalf.json; 121-164 pairs/band).** Sister-half cosine 0.48 / 0.41 / 0.38 / 0.33 / 0.25 vs
     0.00 for other players (centred); the sister is the top-1 match among all other-half adapters 81 / 79 / 77 /
     74 / 50% of the time (chance < 1%). Cross-validated PCA: reliable share of variance 46 / 38 / 37 / 30 / 24%,
     reliable effective dimension 18 / 25 / 23 / 27 / 20 (flat; the raw rise in dimension with rating was noise).
     Answer to 11: style is a continuum of about 20 reliable dimensions at every rating; what shrinks with rating
     is the share of a player's departure from Maia-3 that is stable, not the number of ways to differ. Trait
     reliability (Spearman across halves, 700 -> 2300): castling 0.79 -> 0.48, captures 0.72 -> 0.43, checks
     0.74 -> 0.49, retreats 0.63 -> 0.51, Maia-top-move conformity 0.60 -> 0.43, pawn moves 0.75 -> 0.21; piece-type
     traits are weaker.
   - **12c: does the base model carry the player? (2026-11-19, CPU, knn keys = Maia-3 pooled trunk output, PCA
     128; lab/results/maia_trunk_identity.json).** Mean key over 20 of a side's context games vs the other 20,
     nearest match in pools of 150 sides (chance 0.7%): 40 / 39 / 35 / 37 / 31% top-1 (700 -> 2300). Elo alone
     (mean rating per half) does 41 / 52 / 41 / 55 / 62%, so much of it is the Elo input; with Elo regressed out
     (cubic) the keys still give 31 / 30 / 28 / 27 / 21%, and 15 / 13 / 12 / 12 / 11% from ply >= 30 positions only.
     Maia sees one position + Elo, so this is identity in *which positions a player reaches* (repertoire, then the
     structures they steer into), not the model inferring the player. Function-space adapters by ply (split-half
     ID, ~150 candidates): plies 10-19 89 -> 50%, 20-29 82 -> 42%, 30+ 80 -> 36% (700 -> 2300); identity in how a
     player moves fades with ply but stays far above chance (12e: not only repertoire). Matched data (ctx40 split:
     adapters on ~17 games per half, 100-173 pairs, lab/results/adapter_split_fit0_s200_split.json): adapter ID
     54 / 40 / 38 / 34 / 23% vs trunk keys (Elo regressed out, 20 games, pools of 150) 31 / 30 / 28 / 27 / 21%: on
     equal data the adapter carries more of the player than the positions reached, the margin closing at 2300.
     Reliable dimension 12 / 15 / 13 / 17 / 11 at 17 games vs ~20 at ~97: more games resolve more style axes.
   Where the analogy breaks: per-move player signal is tiny (board constrains moves); no post-training stage in
   human models, so any "RL collapse" lives in the humans (practice, engine prep) and shows only in the Elo curve;
   LLM personas are characters, chess styles are policies (probing/steering transfers, role-play framing does not).

## Done

- 2026-11-12: style_fidelity on v0f_1100/1900/half/quarter (one 40-min job): z's backwards conformity is real at 1500
  (z adds −0.096 [−0.154, −0.040]) and grows with data, absent at 1100/1900; z couples steering and conformity on one
  axis (ρ −0.82). Steering and allocation gains replicate. SCIENCE.md §8.

- 2026-11-11: style_fidelity for v0f (one 13-min job): move-choice steering ρ +0.17 at 3% of the real spread; conformity
  ρ −0.11 (z makes it worse); allocation post 0.58 = history ceiling, z +0.06. `v0e_train eval --style`. SCIENCE.md §8.

- 2026-11-11: policy-filtered backoff (one 90 s job, 866k queries): own near positions beat another player's only in the
  opening (match 0.84 vs 0.49 at JS<=0.01, plies 0-9); from ply 10 own = control; gain past ply 20 is 0.0002. SCIENCE.md §8.

- 2026-11-11: probe_perturb (one 3-min job, 32k pairs): 53% of one-square perturbations change Maia's decision; attack
  heuristic AUC 0.63, Maia mid-layer cosine AUC 0.84, final layer 0.88. Basis for policy-filtered backoff. SCIENCE.md §8.

- 2026-11-11: repeat-visit fidelity metric (one 1-min job over 5 runs), now part of `v0e_train eval`: v0f at 1.2% of ceiling
  with 0.002 nats KL; 98% of repeat visits at plies 0-9; v0f moves the right way but too little. SCIENCE.md §8.

- 2026-11-11: repertoire memory (3 one-minute dump jobs + CPU): a count table, a = 0.5, over v0f gains +0.80 / +0.89 /
  +0.91 nats per repeat visit at 1100 / 1500 / 1900; recency and an adaptive a add nothing. Used as an eval ceiling
  (repeat-visit fidelity), not at inference. SCIENCE.md §8.

- 2026-11-11: v0f at 1100 and 1900 (item 6): gain over teacher −0.0023 / −0.0013 / −0.0006 at 1100 / 1500 / 1900;
  z-swap halves at 1900; repertoire gap holds at every band (z ≤ 1.2% of history gain). SCIENCE.md §8.
- 2026-11-10: evalnet labels for bands 1100 and 1900 train (65 + 75 min) -> `data/evals/v0_{1100,1900}_train_evalnet_ft*.parquet`.
- 2026-11-10: repeat_v0f: on 66,675 held-out repeat visits, v0f's z recovers 1.2% of the personal-history gain
  (+0.011 vs +0.905 nats); ×1.5 per data doubling. Repertoire needs an explicit per-player memory. SCIENCE.md §8.
- 2026-11-10: Maia-3 teacher for bands 1100 and 1900, train (45M + 52M positions, 65 + 73 min) and val ->
  `data/v0/teacher_{1100,1900}_{train,val}`.
- 2026-11-10: fixed-1500 Maia teacher over v0 val (1.39M positions, 2 min) -> `data/v0/teacher_1500_val_e1500`.
- 2026-11-10: v0f_quarter data scaling: z-swap 0.00032 / 0.00069 / 0.00115 at 1/4, 1/2, full; power law near
  N^0.9, not flattening; rating R² flat. SCIENCE.md §8.
- 2026-11-09: v0f_half data scaling: half the games costs 40% of z-swap gain and 30% of the teacher gain;
  re-ID at 20+ games and rating unchanged. Bigger corpus helps z a little. SCIENCE.md §8.
- 2026-11-09: player_markov (CPU): personal history at repeated positions beats Maia by 0.93 nats per visit;
  per-player gain split-half ρ +0.89. SCIENCE.md §8.
- 2026-11-09: style traits at scale on the net-labeled v0 corpus: 26,757 players; steering +0.49 and
  allocation +0.51 split-half, performance null within ±0.015. SCIENCE.md §8.
- 2026-11-09: evalnet labels for all 700k v0 train games (49M positions, 70 min) ->
  `data/evals/v0_train_evalnet_ft.part*.parquet`.
- 2026-11-09: evalnet fine-tune on 100k games × 3 epochs (99 min). MAE 2.78 → 2.62 win%, blunder recall 0.59 →
  0.62, still short of 1,000-node Stockfish (0.65). Checkpoint `data/probe/evalnet_ft_g100k_e3.pt`. SCIENCE.md §3.1.
- 2026-11-09: evalnet labels for all 36k corpus games (3.5 min). Net T2+T1 on 36k matches Stockfish T2+T1 on 12k
  (214 vs 214). SCIENCE.md §3.1.
