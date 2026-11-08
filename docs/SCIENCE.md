# Doppel science program

This file defines what Doppel models about human chess behavior and how each capability is measured.
Results live in `lab/results/*.json`; the numbers quoted below are copied from those files with their dates.

## 1. Scope: the model of human behavior

A Doppel plays like one human. We split "plays like" into **capabilities**, each a prediction about what a human
does at a decision point. The set of capabilities *is* our model of human behavior; anything outside it is
assumed to be inherited from the move-choice teacher or ignored.

| Capability | Question | Status |
|---|---|---|
| Think time | How long does this player spend on this move? | Modeled (§5) |
| Resignation | Does this player resign now instead of moving? | Modeled (§6) |
| Blunders | Does this move throw away a large share of the player's winning chances? | Adopted, no results yet (§7) |

Move choice itself is a research target, but at this time is only planned future work. Currently we use
off-the-shelf models like Maia, and plan to leverage other chess-playing models such as ChessMimic when we begin
modeling move choice. Until then, v0 distils move choice from Maia-3 (`doppel_lab/teacher.py`) and we check only
that it does not regress (top-1 and perplexity, `doppel_lab/top1.py`).

Blunders are the first capability that scores move choice itself. Every move model implies a blunder
probability, the policy's mass on blundering moves, so blunders compare all the machines in §4 on how human
their mistakes are, without changing what v0 trains.

## 2. The benchmark rule

Every capability is scored against three kinds of reference, on the same held-out human games, with the same
metric:

1. **Trivial prior**: the base rate, or the mean. A model that cannot beat this has learned nothing.
2. **Heuristics**: hand features (§3) combined by a simple fitted model (ridge or logistic regression). This is
   the number a neural head must beat to justify its cost.
3. **Other chess machines**: existing engines or human-imitation models that make the same prediction, or the
   closest proxy they have (§4).

Where possible we also report a **human noise floor**: the same statistic computed between two halves of the human
data, so "as good as human-vs-human" is a visible ceiling.

Rules that keep the comparison fair:

- Held-out games only (`doppel_lab/split.py`); the same positions for every model being compared.
- External human data (the original Allie paper's 2022 blitz test set, `doppel_lab/allie.py`) is a second test
  set. Allie the model is also a machine (§4); the two uses are separate.
- One rating band at a time; v0 work is the 1500 band.
- Every source is adjudicated the same way (`doppel_lab/metrics.py`): a game "ends" at the first ply where
  |eval| ≥ 600 cp holds for 6 consecutive plies, because bots never resign and humans do.
- A result is a JSON file under `lab/results/` written by a script in `lab/`, never a number typed into a doc.

## 3. Features: compute tiers and information regimes

Every "dumb" baseline draws from one canonical feature set, so a new baseline is a new estimator, not new
features. Each feature carries two labels: a **compute tier** (what it costs) and an **information regime** (what
it is allowed to know). A baseline declares the highest tier and regime it uses, and results tables report both.

**Compute tiers.** A feature's tier is the highest tier among its inputs; a derived feature inherits it (Stockfish
volatility is T2 even though the rolling std is free).

| Tier | Features | Cost (lab server) | Coverage |
|---|---|---|---|
| **T0** free | Clocks, increment, think time, ply, both ratings, legal-move count, in check, move type, **material** (P 1, N 3, B 3, R 5, Q 9) | microseconds per position | full corpus, on the fly |
| **T1** Maia-3 | Policy entropy, top-move probability, played move's log-prob and rank, win/draw/loss | ~12,000 positions/s on the GPU (79M model, through the warden); ~3 s per game on the CPU | full corpus |
| **T2** Stockfish | cp, winning chances, CPL, wc loss, Lichess move class, at 50,000 nodes | ~49 ms per position per core (~depth 16) | label sets (§7); full corpus ~24 h on 28 cores |
| **T3** deep | Stockfish on every legal move (blunder potential β), 1M-node spot checks (~1 s per position), tablebases | hours per 2,000 games | eval sets only |

Costs measured 2026-11-06 with Stockfish 16 (about 2 M nodes/s per core) and the Maia-3 teacher job.

**Information regimes.** All three are causal: a feature for ply k never reads ply k+1 or later.

| Regime | May read | Status |
|---|---|---|
| **R0** point | Position k and game metadata (ratings, time control). No earlier plies. The original Maia's board-only setup. | built (analyzer channels) |
| **R1** trajectory | Every ply of this game up to k: the causal transforms of §3.4. A whole-game summary exists for game-level reports only and is never a predictor. | built (analyzer transforms) |
| **R2** career | Every game by this player that started before this game. | design only |

Book keys come in both regimes: the position (Zobrist) key is R0, the move-order prefix key is R1 (§8, Book depth).

R2 is the only fair baseline for a player-personalized Doppel, since such a Doppel has seen the player's history.
Our split is player-disjoint (`lab/doppel_lab/split.py`: a player is held out iff `sha1(name) % 5 == 0`), so a
held-out player's history exists only in the held-out games. R2 therefore needs a within-player time split:
earlier games are history, later games are targets. A personalized Doppel needs the same split, so the baseline
and the Doppel see the same data.

**Fair comparison.** A machine is scored against baselines at or below its own inference cost and information.
The v0 Doppel at inference is T1/R1: a Maia-3 trunk plus history and clocks. T2 and T3 baselines are **oracle**
baselines: they know the engine's verdict, which neither the human nor the Doppel has. They are ceilings for
dumb estimators and are labeled as oracles.

**Timing.** Each feature is also labeled by when it is known relative to the move at ply k: **pre** (before the
move is chosen: position k, the clocks, history through ply k−1), **act** (the move itself and how long it took:
this move's think time, its Maia-3 log-prob) or **post** (needs the next position: CPL, wc loss, blunder). Post
features are targets, never predictors of the same move. Every history transform comes in two forms: `x_ewma`
includes the current move and inherits its timing; `x_ewma_prev` is its value as of the mover's previous move and
is always pre.

**Storage and code** (`lab/doppel_lab/features/`):

- `registry.py` defines every feature once: name, tier, regime, timing, table (per move from the mover's view, or
  per position from White's), and `VERSION` (now `f1`). `select(tier, regime, timing)` lists what a baseline may
  use. A test fails if the analyzer or the table builder emits a name the registry lacks.
- `channels.py` and `transforms.py` compute the features for one game. The analyzer and the corpus tables both
  call them, so the HTML profile and the baselines cannot drift.
- Only raw channels that cost something are stored as sidecars joined on `game_id`: the Maia-3 teacher shards
  (T1) and `evals_*` (T2).
- `table.py` builds per-move tables from those sidecars into `data/v0/features/moves_<VERSION>_<tier>_<band>_<split>/`.
  The tables are a cache that can be rebuilt (about 9,000 moves/s on 8 cores: val in about 3 min, train in about
  1.5 h). A T2 table covers only games with evals.
- Not stored yet: Maia-3 win/draw/loss for v0 positions, so the T1 position features (expected-score volatility,
  sharpness) exist only in the analyzer.

### 3.1 Stockfish eval

- **What it is:** an evaluation function. Stockfish searches ahead (alpha-beta with pruning, a transposition
  table, and an NNUE neural net as the static eval at the leaves) and reports a score in **centipawns** (cp),
  hundredths of a pawn, from White's view. Forced mates are stored as ±10,000. NNUE has been the main eval since
  Stockfish 12 (2020); the handcrafted eval was removed in the 16 cycle, documented in 16.1 (February 2024). We run
  Stockfish 16 (Debian package).
- **Winning chances:** centipawns are converted with Lichess's logistic curve
  `wc(cp) = 2 / (1 + exp(-0.00368208 · cp)) − 1`, range −1 to +1. The constant was fitted on Lichess games, so it
  describes human players. Per-move losses are measured in wc, not raw cp, so a drop from +800 to +500 is not
  scored like a drop from +100 to −200.
- **Cost:** our 12,000-game human eval set took 24 minutes on 28 cores (`lab/doppel_lab/analyze.py`). The full
  v0 train set (48.7 M positions) would take about 24 hours on 28 cores.
- **When it is worth it:**
  - Yes: evaluation sets, labels that need ground truth (blunders), small analyses, spot checks of a model.
  - No by default: per-position training features at corpus scale. Use material or Maia-3 win/draw/loss first,
    and add Stockfish only when an ablation on a sample shows it earns the day of CPU.
  - Never at serve time without a separate decision: a feature the model needs as input must also be computed
    when the Doppel plays.
- **Cheaper Stockfish (measured 2026-11-06, 40 val games, 2,866 positions, scored against 50k-node labels):**

  | Budget | Cost vs 50k | wc correlation | wc MAE | Blunder recall | Blunder precision |
  |---|---|---|---|---|---|
  | static NNUE eval, no search | — | 0.878 | 0.131 | 0.06 | 0.33 |
  | 100 nodes | 1% | 0.935 | 0.100 | 0.51 | 0.40 |
  | 1,000 nodes | 4% | 0.985 | 0.056 | 0.65 | 0.78 |
  | 5,000 nodes | 13% | 0.993 | 0.036 | 0.80 | 0.86 |

  The neural part is not what costs: an NNUE eval takes microseconds, and the cost is the search, tens of
  thousands of evals per position. The static eval misses 94% of blunders, because a blunder is usually a tactic
  that only search sees. Search is sequential alpha-beta built around incremental CPU updates of the net, so a
  GPU does not speed it up; corpus throughput already scales linearly with cores (one engine per game). Labels
  stay at 50k nodes. A **T2-lite** channel at 5,000 nodes (full corpus in about 3 h on 28 cores) is an option
  when a baseline needs Stockfish features on every position. **Planned (future ML work): an auxiliary eval net**, trained
  on our 50k-node labels to predict wc and wc loss directly, batched on the GPU at T1 cost. It is a new feature
  with its own measured error against the 50k-node labels, not a faster Stockfish. Blunder rates from 2,866 positions are noisy (about ±0.05).
- **Caveat:** Stockfish sees tactics a 1500 player does not. A better eval is not always a better predictor of
  human behavior; Maia-3's win/draw/loss is closer to how a human of that rating sees the position.

### 3.2 What is already computed

| Artifact | Contents | Coverage |
|---|---|---|
| `data/v0/games_1500_{train,val}.parquet` | moves, clocks (mover's clock after each ply, whole seconds), ratings, result, termination | 700k / 20k games |
| `data/v0/teacher_1500_{train,val}/shard-*.npz` | Maia-3 top-16 moves and log-probs, played-move log-prob, entropy, logsumexp | 48.7 M / 1.39 M positions |
| `data/v0/resign_1500_{train,val}.parquet` | game ending class, resigner, resigner on move, plies | all v0 games |
| `data/human/*eval*.parquet` + Stockfish evals | per-ply cp at 50k nodes | the 2026-10 eval sets |

Not stored yet: Maia-3 win/draw/loss for v0 positions (about 70 GPU-minutes through the warden).

### 3.3 Derived heuristics

Built from the features above and used across capabilities:

- **Centipawn loss (CPL)** and its game average **ACPL**: `max(0, eval before − eval after)` from the mover's view,
  evals clipped to ±1000 cp.
- **Blunder**: a drop of ≥ 0.3 in winning chances on one move (Lichess's threshold). Smaller Lichess classes:
  mistake and inaccuracy.
- **Realized shock** (`lab/scripts/shock.py`): an oracle-free material loss. The opponent's move leaves the
  player's material lead ≥ 3 below where it stood before the player's previous move, and the deficit persists for
  the player's next two moves. A blunder is not a shock: at low ratings many blunders are never converted.
  The think-time probe uses a stricter version: a loss of ≥ 9 (a queen) that persists for 2 own moves.
- **Clock fraction**: remaining time over base time; **think fraction**: `logit((think + 0.5) / (remaining + inc))`.
- **Tempo**: the player's running median and decayed mean of think fraction so far in the game.
- **Policy confidence**: Maia-3 top-move probability and entropy. Used by the production think-time heuristic.

### 3.4 Per-game trajectories (the game analyzer)

Every per-position feature above is also a **trajectory**: one value per ply over the whole game. Simple causal
transforms of a trajectory give history features without a sequence model. The analyzer
(`lab/doppel_lab/analyzer/`) computes them for one game and renders an HTML page with every trajectory on a
shared ply axis:

    uv run python -m doppel_lab.analyzer <lichess game ID | file.pgn | v0 game_id> [--tables DIR]

It reads finished games only, and refuses a game with no result. One 130-ply game takes about 9 s on the CPU:
Maia-3 about 3 s, Stockfish at 50,000 nodes about 5 s. Nothing is served; the page is a local file.

- **Channels** (raw, per ply): clocks, think time and think fraction, legal moves, check, material
  (T0); Maia-3 win/draw/loss, expected score, sharpness, policy entropy, top-move probability, played move's
  log-prob and rank (T1); Stockfish cp, winning chances, CPL, Lichess move class (T2).
- **Transforms** (causal: a value uses only the current and earlier plies):
  - Exponential moving average with α = 0.3, as in the probe's decayed tempo.
  - Rolling mean and standard deviation over the last 5 own moves.
  - Velocity: the change from the previous own move.
  - Running median of think fraction before the move (tempo).
  - On the position series: **volatility**, the rolling standard deviation of win % over Lichess's window (game
    length / 10, clamped to 2..8); velocity; and each side's **drawdown** below its running peak.
- **Events:**
  - Lichess move classes: blunder, mistake and inaccuracy, at win-chance drops of 0.3, 0.2 and 0.1.
  - Realized shock, as defined in §3.3.
  - Entry into time trouble: own clock below 10% of base time.
  - Lead change: win chances crossing from beyond ±0.2 to beyond the opposite ±0.2.
  - Game end, classified by `doppel_lab/resign.py`.
- **Summary:** one row per side, the per-game "dumb features" for simple estimators. It covers ACPL, move
  classes, median think time, tempo drift (second-half minus first-half mean think fraction), time-trouble moves,
  Maia-3 top-1 rate and surprisal, mean volatility, worst drawdown, mean sharpness and shocks. `--tables` writes
  it as parquet next to the per-ply tables.

**Finding, sharpness.** jk_182's formula (§10) depends on the draw probability: `(2 / log((1−W)(1−L)/(WL)))²` grows
without bound as D → 0. At 1500 blitz Maia-3 rarely gives a draw more than a few percent, so balanced positions
score above the plot cap of 50, while lopsided ones score low. It was designed on Lc0's win/draw/loss for strong
players; for our bands it needs a variant before it is used as a feature.

## 4. Chess machines we benchmark against

| Machine | What it predicts | Used for |
|---|---|---|
| **Maia-3 79M** | Human move distribution by rating; win/draw/loss | Move-choice teacher; outcome calibration; frozen features for the think-time probe |
| **ChessMimic** | Moves, a clock model (think-time distribution), a winner model | Think-time and outcome baselines |
| **Stockfish 16** | Best play and eval | Labels and eval features only; it does not imitate humans |
| **Allie** (original, ICLR 2025) | Moves, think time, resignation (a `<resign>` token), value; 355M decoder, blitz 2022 | Think-time and resignation baseline; published numbers only so far |
| **Allie 2.1** (`yimingzhang/allie-2.1`, MIT) | Moves, think time (63-bin head), win/draw/loss; reads both clocks and the previous think time; all time controls | Strongest machine for both capabilities and for move choice; ~18 GB, so a warden job |
| **`heuristic_think_ms`** (`packages/doppel_model/doppel_model/think.py`) | Our production think-time placeholder: a hand-set rule on clock, ply, legal moves and Maia-3 top-move probability, with lognormal noise; nothing fitted | Must be benchmarked like any other machine (§5) |

Only Allie resigns. The original Allie resigns through its `<resign>` token; Allie 2.1's bot resigns through a
hazard fitted on its own win/draw/loss, ply, rating, time control and clock (`src/allie/lichess/README.md` in the
Allie repo). Other machines are scored through proxies (§6).

Allie 2.1 also matters for move choice: on the Allie repo's protocol it beats Maia-3 79M by 0.042 nats per move
and 1.2 points of top-1, and unlike Maia-3 it reads the clock. That is a candidate teacher for later versions,
not a change to v0.

## 5. Capability: think time

**Target.** Per own move, `think = clock[k−2] − clock[k] + inc` in seconds (Lichess clocks are whole seconds).
Two scales are reported: `log1p(think)` and think fraction (§3.3).

**Metrics.** R² on held-out positions for both scales; Pearson r on log think time; the shock residual (mean
actual − predicted on own moves 1 to 5 after a realized shock).

**Results** (`lab/results/think_probe_2026-10.json`, ridge on 190k positions, 58k test):

| Model | R² log1p(s) | R² think fraction |
|---|---|---|
| Clocks only (heuristic) | 0.167 | 0.483 |
| Clocks + tempo (heuristic) | 0.268 | 0.550 |
| Maia-3 features, current board | 0.407 | 0.626 |
| Maia-3 features, 8-ply history | 0.442 | 0.646 |
| Maia-3 features, history + tempo | 0.450 | 0.651 |

ChessMimic's clock model (`lab/results/guardrails_eval.json`): Pearson r on log think time 0.54 to 0.58 across
bands, 0.56 at 1500.

**Shock** (`lab/results/shock_2026-10.json`): after a realized shock, humans' log think time drops by 0.195 on
average over their next 5 moves (95% CI 0.187 to 0.204, 8,000 events). ChessMimic self-play, which has no
internal state, shows a similar average drop (0.197), so on this average the effect is explained by the position
and clock. On the first move after a queen-loss shock (619 test events) the probe models with Maia-3 features
under-predict human think time by 0.09 to 0.11 in log1p(s): humans take longer than a stateless prediction.

**Allie.** Not yet run on our data; the published numbers are not comparable to the table above:

- Original Allie: Pearson r = 0.697 between predicted and human think time. The head is trained with squared
  error on raw seconds, so r is on raw seconds, where long thinks dominate; ours is on log seconds. The test set
  is 2022 blitz, all ratings, and drops the first 5 moves and every move with under 30 s on the clock. The model
  under-predicts long thinks.
- Allie 2.1 reports distributions, not correlations: on its July 2026 held-out games the bot's sampled think
  times match human medians and 90th percentiles per time control (blitz: median 2 s human vs 2.4 s bot, 90th
  percentile 11 vs 10.8).

Both kinds of score belong in our table: R² and r on log seconds for prediction, and distribution distance for
sampled think times.

**Time management by rating** (`doppel_lab/clock_curve.py`, 2026-10 corpus, 3+0; clock left after own move N,
as a share of the base). Weak players spend early: after move 20, a 600 has 49% left and a 1800 has 71%. From
1800 up the early curve flattens (93% after move 10, from the book). Deep into long games, strong players run
lower: after move 50, 10–11% left at 2200–2600 against 17–18% at 1400–1600, and 47–48% of 2400+ sides are
under 10% of the base. Weak players are short of time from the start; strong players save time early and spend
it in long, complex games. The pattern is U-shaped in rating, so a think head conditioned on rating alone must
not be monotone.

**Think time changes which move is played** (`doppel_lab/think_policy.py`, 1500 train table, clock ≥ 10%).
At fixed Maia-3 entropy, long thinks move humans away from Maia's top choice. In the middle entropy quintile,
the played move is Maia's top-1 62% of the time at 1 s, 59% at 2–3 s, 51% at 4–10 s and 42% at 11 s+. In the
easiest quintile it runs 95% → 71%. The same holds among non-blunders (63% → 45% in the middle quintile), so a
slow move is often a different, considered move, not just a worse one. Move and think time are therefore not
independent. A Doppel that samples think time and move separately will play long thinks too intuitively.
Condition the policy on the sampled think time, or sample the two jointly. v0's residual policy sees only
the board, z and the ratings (`v0_model.py`), so it can't do this yet.

**Long thinks cluster** (`doppel_lab/think_ac.py`, 6.0 M own moves, 1500 train table). Maia-3 entropy and
top-move probability, clock, ply and the side's own pace in the game explain R² 0.30 of log think. The rest
is autocorrelated: residual correlation with the side's previous move is +0.153, then +0.143 at lag 2, +0.114
at 3, +0.068 at 5 and 0 by 10. Players have short "slow" and "fast" spells lasting about five moves, which no
position feature explains. A think-time head needs the recent think history to reproduce them. v0's head has it: `tempo_feats`
in `v0_data.py` carries the median and a 0.7-decay EWMA of the side's recent think fractions. A check that
sampled v0 times show the +0.15 lag-1 correlation is still open. Sampling each move's time independently would
flatten them.

Players play the book from memory (`doppel_lab/book_think.py`, 87.7k train games with a book exit at plies 8–30,
plies relative to the exit, both sides). Think time rises slowly through the book, from 2.0 s six plies before
the exit to 2.9 s one ply before, then jumps to 3.9 s on the leaving move. The opponent's reply to the new
position takes 3.7 s, against about 3.0 s on the trend. Maia-3 entropy rises at the exit too (1.68 vs 1.45), so
part of the jump is a harder position. Book depth (the T1 book features) belongs in the think-time head's
inputs. Without difficulty controls (`think_ac_synth.py`, 3+0, the side's mean removed),
human 1500 sides show +0.157 at lag 1 and +0.083 at lag 5. ChessMimic's self-play shows +0.208 and +0.095,
spells included and slightly too sticky. v0d self-play does not yet record clocks, so its spells can't be
checked. Recording sampled think times in `selfplay.py` is a prerequisite for a timing check in the gate.

**Open.** `heuristic_think_ms` and Allie 2.1 have not been scored on held-out human games yet. They are the first
baselines the v0 think-time head has to beat on the same table.

## 6. Capability: resignation

**Target.** For the side to move at each decision position 0..plies: 1 if they resign in that position instead
of moving, else 0. Labels from `doppel_lab/resign.py`: a game with termination "Normal" that is neither checkmate
nor a draw is a resignation by the loser. Final positions of games that did not end in resignation are dropped.

**Label facts** (train, 700k games): resignation 50.0%, mate 24.5%, time loss 22.3%, draw 3.2%. The resigner was
on move in 88% of resignations; the other 12% resign during the opponent's turn and are out of scope for v0.

**Metrics.** AUC (does it rank lost positions above others), average precision (does it pick the moment),
log-loss gain over the prior, and, from v0 on, per-game timing error: how many own moves the predicted resignation
is from the real one.

**Results** (`lab/results/resign_baseline_1500.json`, logistic regression, 4.2 M train positions from 60k games,
1.4 M val positions, positive rate 0.64%):

| Model | AUC | Avg precision | Log-loss gain |
|---|---|---|---|
| Prior | 0.50 | 0.006 | 0 |
| Hand features: material, clocks, ratings, ply, legal moves, check | 0.881 | 0.051 | 18.7% |

Reading: the hand features tell who is lost; they do not tell when the player gives up.

**Machines (to do).** Scored on the same table as the hand features:

1. **Original Allie.** It resigns when the `<resign>` token outscores every legal move and its value is below
   −0.9. Published: true positive rate 86.4% and false positive rate 0.1%, per position, on 2022 blitz. Those
   are per-position rates on a different test set and filter, so they must be rerun on our val set before they
   are compared with our AUC and average precision.
2. **Allie 2.1's resign hazard.** Fitted on Allie's own win/draw/loss, ply, rating, time control and clock. On
   its held-out games humans resign in 22.5% of player-games and the hazard in 14%; P(loss) at resignation
   matches humans at the 25th percentile and median (0.86, 0.95). This is the strongest resign baseline.
3. **Proxies for machines that do not resign:** the adjudication rule (|eval| ≥ 600 cp for 6 plies, needs
   Stockfish), and a threshold on Maia-3 win/draw/loss.

Allie's data agrees with ours on one label fact: 87% of its humans resign on their own turn (ours: 88%).

**Known gaps.** Off-turn resignations (12%). Some time losses are disguised resignations, where a player in a lost
position stops moving; v0 counts them as negatives.

## 7. Capability: blunders

**Why.** Playing like a human includes making that human's mistakes, in the same kinds of positions. It also
extends the think-time work directly: longer thinks predict more blunders, in every slice Anderson et al. looked
at (§10). And it unlocks comparisons. Think time and resignation have few machine baselines, but every move model
implies a blunder probability, so every machine in §4 can be scored on the same table.

**Target.** For each own move, from the mover's view: 1 if the move drops their winning chances (Stockfish at
50,000 nodes, §3.1) by ≥ 0.3, else 0. This is Lichess's blunder threshold, the class the analyzer already labels
(`doppel_lab/analyzer/channels.py`). Every result is also reported at ≥ 0.2, which equals Maia's unit (a drop of
10 points of win probability), so the numbers compare with the Maia paper. All positions are kept, including
time trouble and lost positions; a nearly lost position can barely drop 0.3, and the prior absorbs that.

**What v0 changes.** Nothing in training. Blunders score the move-choice part of a Doppel as a machine. A
dedicated blunder head is a baseline only; it would not change how a Doppel plays.

**Metrics.**

1. AUC, average precision and log-loss gain over the prior, as in §6.
2. Calibration: predicted P(blunder) against the observed rate in 10 bins (a reliability curve), and the expected
   calibration error. A machine can rank well and still blunder twice as often as humans, and we need to see that.
3. Rate match: mean predicted P(blunder) against the human rate, overall and by band, clock fraction and
   difficulty bin. This is the number a Doppel has to match.
4. Human noise floor: on positions humans faced at least 10 times, as in Maia's collective task, how well one
   half of those humans' blunder rate predicts the other half's.

**Scored against.**

1. **Prior:** the blunder rate per band.
2. **Heuristics, before the move:** material, clock fraction, time trouble, both ratings, legal moves, check,
   tempo (T0); plus Maia-3 policy entropy and top-move probability (T1) as a cheap difficulty proxy.
3. **Heuristics, joint with the clock:** the same features plus this move's think fraction. This variant is not a
   predictor a Doppel can use before moving; it measures the think-time and blunder link on our data.
4. **Blunder potential** β, Anderson et al.'s difficulty measure: the fraction of legal moves that blunder, from
   Stockfish on every legal move (T3, eval sets only). A policy-weighted version, β weighted by Maia-3's move
   probabilities, is a second variant.
5. **Machines, policy-implied:** P(blunder) = the sum of a policy's probabilities over the moves that blunder.
   Maia-3, Allie 2.1, and the v0 Doppel, with z from the prior and from the posterior. A strong engine's best move
   almost never blunders, so Stockfish gives the floor of a machine that plays perfectly by its own measure.

**Data and cost.**

- Training and heuristic labels need one eval before and after each ply, the `doppel_lab.analyze` pass: about
  3 hours on 28 cores for 100k games (6.5 M positions).
- Blunder potential and the policy-implied scores need a Stockfish score for every legal move, so they run on an
  eval set of 2,000 val games (about 130k positions, about 30 legal moves each). A search over all legal moves
  at once (MultiPV) is the plan. The node budget is set so the best move's score matches the single-line 50k-node
  eval, and the cost is measured on 100 positions before the full run; the estimate is 1 to 4 hours on 28 cores.
- Optional exact labels: in endgames with at most 5 pieces, a Syzygy tablebase (about 1 GB) gives the
  game-theoretic result, as in Anderson et al. That subset checks how noisy the Stockfish labels are.

**Known gaps.**

- A Stockfish blunder is not always a real loss: at 1500 many blunders are never punished (§3.3, realized shock).
  The target measures error against the engine, not against the outcome.
- Labels at 50k nodes miss deep tactics that a 1M-node search finds. T3 spot checks at 1M nodes measure how often.

**Results (val, 1500 band, 5-fold CV over games; 1.39 M moves, blunder rate 6.47% at ≥ 0.3).**
`lab/results/blunder_baselines_1500_cv.json`, gradient-boosted trees per feature set:

| Features | AUC | Avg. precision | Log-loss gain |
|---|---|---|---|
| Prior | 0.50 | 0.064 | 0 |
| T0 (board, clock, ratings) | 0.731 | 0.137 | +0.092 |
| T1 (+ Maia-3 entropy, top-move prob.) | 0.755 | — | +0.113 |
| T1 + this move's think fraction | 0.761 | — | +0.118 |
| T2 (+ Stockfish eval before the move) | 0.847 | 0.306 | +0.229 |

All sets are calibrated (ECE ≤ 0.0014). Players below 1450 are under-predicted (observed 0.074, predicted
0.067).

Final fit on train (100k games, 6.98 M moves, blunder rate 6.47%), scored on val
(`lab/results/blunder_baselines_1500_train.json`): T0 AUC 0.734 (AP 0.139, log-loss gain +0.095), T1 0.760
(+0.117), T1 + think 0.766 (+0.123), T2 0.850 (AP 0.312, +0.235); ECE ≤ 0.0010 for all. At ≥ 0.2: T0 0.723,
T2 0.828. The CV numbers held, so these are the blunder bars.

Think time and blunders at fixed difficulty (`doppel_lab/think_value.py`, train table, ply ≥ 10, clock ≥ 10%):
raw think quintiles run from a 4.2% to an 11.2% blunder rate, the confounded view. Within Maia-3 entropy deciles,
moves played much slower than the player's own pace in that game still blunder 3.8 pt more, and much faster
moves 2.2 pt less. A long think marks trouble the player senses beyond what Maia's entropy captures; it does not
fix it, which is why think time adds to the T1 blunder baseline. Sides whose think time tracks difficulty
(Spearman of think vs entropy across their moves, quintiles −0.08 → 0.51) blunder a little less (8.1% → 7.4%),
with the gain flat from the middle quintile on.

Instant moves are the safest (`doppel_lab/premove.py`). Moves played in 0 s (4.9% of moves; premoves and
instant replies) blunder 3.5% of the time, against 4.4% at 1 s, 6.1% at 2–3 s, 9.2% at 4–10 s and 12.8% at 11 s+.
That holds inside every Maia-3 entropy quintile, including the hardest (5.4% at 0 s, 12.8% at 11 s+). Players
premove only when the reply is obvious (Maia-3 top-move probability averages 0.72), so a Doppel should not add
a premove blunder risk.

Blunders cluster (`doppel_lab/blunder_chain.py`; rates compared inside cells of Maia-3 entropy decile x the
mover's win chances x clock fraction). Right after the opponent blunders, the mover blunders 29.9% of the time
against 5.6% otherwise (+22.8 pt matched, z = 337). A position that is winning because of a fresh blunder
usually needs one exact reply, and missing it counts as a blunder, so this is mostly "fails to punish": 1500
players miss the refutation about 30% of the time. After their own blunder, players blunder again 16.4% vs
6.7% (+9.8 pt matched), partly because an unrescued hanging piece stays hanging. For a Doppel, both are
targets worth matching: a 1500 Doppel that always punishes blunders is not playing like a 1500.

Move types matter less than position (`doppel_lab/blunder_kind.py`, 1.8 M own moves, 30k train games, base
rate 7.45%). Queen and knight moves blunder 1.21x and 1.16x as often as the average move, king moves 0.67x.
Captures blunder 0.79x, checks 0.82x and moves right after the opponent's capture 0.86x, since forcing moves
are easier. A piece left on an attacked, undefended square (a crude static test) blunders 1.29x. Escaping an
attacked piece changes nothing (0.99x). None of these lifts comes close to the position-level features above.

The most common early blunders at 1500 (`doppel_lab/traps.py`, plies 4–24, `lab/results/traps.json`) are
nearly all f7/f2 tactics in open games. 4.Ng5? in the Scotch (1.e4 e5 2.Nf3 Nc6 3.d4 exd4) appears 105 times.
3...Bc5? in the Petrov-Italian (1.e4 e5 2.Nf3 Nf6 3.Bc4) is played in 32% of visits, and is a blunder because
of Nxe5. Then come the early Bxf7+ shots. Several entries are missed punishments. After 1.e4 e5 2.f4 d6
(or Nc6) 3.fxe5?, 3...Qh4+ wins (−279 cp at depth 24), and the natural recapture, played in 89–100% of visits,
throws that away. Depth 24 confirms 3...Bc5? (+155 for White) and 4.Ng5? (−140). These are candidate lesson
content for the client ("punish this"), but the labels come from 50k-node evals, so each line needs a check at
depth before it ships.

The machine check (`doppel_lab/punish.py`, 1,500 val positions right after an opponent's blunder plus 1,500
matched controls; Stockfish at the label budget on every legal move; Maia-3 79M at 1500 vs 1500): after a
blunder, 84% of legal moves fail the label rule (only the refutation keeps the gain). Humans fail 30.9% of the
time. Maia-3's policy-implied rate is 30.4%, but its top-1 move fails only 18.3%. In matched calm positions:
human 6.5%, Maia-3 implied 7.9%, top-1 3.8%. So sampling from Maia-3 reproduces 1500 humans' missed
punishments, and argmax play does not. This is the first policy-implied machine score for §7 item 5.

On 6,000 random val positions (ply ≥ 2, blunder rate 6.75%), the policy-implied P(blunder) is the strongest
pre-move predictor yet: AUC 0.932, AP 0.541 (0.928 / 0.527 on a rerun, since Stockfish's per-move results vary a little between
runs). Blunder potential β (the share of legal moves that blunder) scores
0.824 (AP 0.217), and Maia-3's entropy and top-move probability alone score about 0.60. The T2 table baseline
reaches 0.850. Mean implied rate 7.6% vs 6.75% observed: Maia-3 at 1500 slightly over-blunders. The cost is
Stockfish on every legal move (about 0.1 s of CPU per position at 50k nodes, 24 workers), so it suits labels
and evaluation, not live play.

When 1500 players resign (`doppel_lab/resign_when.py`, 100k train games with stored evals; half end in
resignation): the median resigner stands at win chances −0.87, and 70% resign at −0.8 or worse. Played on,
1500 players save those positions surprisingly often. From (−0.95, −0.8] the side standing there later draws
or wins 15% of the time; from (−0.8, −0.6] 26%. So a resignation at −0.87 gives up about a 15% chance of a
draw or better, mostly through the opponent's later blunders and the clock. About 3% resign at −0.2 or better,
which are mostly sides short of time or abandoning games. The resign head should copy this distribution; it
should not resign at engine-hopeless positions only.

The clock is the main reason to play on (`doppel_lab/flag_hope.py`, 104k lost positions, win chances ≤ −0.6, 3+0
and other blitz): with equal clocks, the losing side saves 19.6% of the time. A lead of over 30% of the base
raises that to 38.3%, and 25.4% of positions are won on time outright. Trailing by over 30%, the save rate is
7.5%. Saves on the board (the opponent blunders back, or abandons) stay at 7–13% whatever the clocks. The resign
head should see the clock difference, not only the eval. v0's does: `clock_feats` gives it both clocks. Policy-implied machine
scores (item 5) and blunder potential (item 4) need the MultiPV pass, not run yet.

## 8. Game-level, book, population and era baselines

**Why.** These are the surface-feature scores a learned match encoder (Maia embeddings, the Doppel z) has to
match or beat on the same splits (§2). Every result here is CPU-only.

**Game level** (`doppel_lab/game_science.py`, 36k human games, `lab/results/game_science.json`). From 1100 to
2100: eval volatility 4.49 → 3.72, ACPL 69.8 → 46.4, blunder rate 8.0% → 4.6%, first blunder at move
11.6 → 19.9. Lead changes stay flat at about 2.3 per game. Stronger players move faster (median think
3.6 → 2.4 s) but reach time trouble more (3.2% → 5.7%). Rating from one game: MAE 281 (time control only),
240 (T0), 223 (T2), with strong shrinkage to the mean.

**Book depth** (`doppel_lab/book.py`, `book_science.py`; 22.7 M train games). Of 1.10 B distinct move-order
prefixes, 98.5% occur once. Median ply of the first never-seen prefix: 9 at 800, 12 at 1500, 13 at 2400.
Weird-move rate (a move chosen by < 1% of the ≥ 100 games reaching that prefix): 6.0% → 2.5% → 1.4%. Depth
grows about 0.5 ply per doubling of the corpus. These are R1 features: the move-order key reads the path.

Position book (`lab/poskey`, `doppel_lab/posbook.py`, `lab/results/position_book.json`; same 22.7 M train games
and 170k heldout games). The key is the Polyglot Zobrist hash of the position (R0), equal to
`chess.polyglot.zobrist_hash`. The month has 1.03 B distinct positions in the first 80 plies, and 97.5% occur
once. Counting takes 54 s on 26 threads (430k games/s) at 8.1 GB peak RSS, with a 1.3 GB Bloom filter in front
of a 26.6 M-key exact table. Merging transpositions adds 2 to 5 plies of book depth, more for stronger players:

| Band | Book exit (R1, move-order key) | Book exit (R0, position key) | Weird (R1) | Weird (R0) |
|---|---|---|---|---|
| 800 | 9 | 11 | 6.0% | 6.3% |
| 1500 | 12 | 15 | 2.5% | 2.7% |
| 2400 | 13 | 18 | 1.4% | 1.4% |

Book exit is the median ply of the first position never seen in train. The 1M-game subsample's +2 plies was
the small-corpus end: the gap grows with corpus size, because position depth gains 0.71 ply per doubling
against 0.51 for move order. The weird-move rate barely changes. Weird (R0) is a move chosen by < 1% of the
≥ 100 train visits to the parent position. So transpositions lengthen the book but don't make played moves
look more expected. R1 residual: 3.7% of plies at a common position (≥ 100 by position key) have a rare
move-order prefix (< 2). That is an unusual route to a common position. It rises with rating, from 2.3% at 800
to 4.4% at 2400, and 17% of games do it at least once.

The Bloom filter never undercounts. On 1M games, against an exact count, 0.035% of repeated positions read one
too high. At 1% FPR, though, 6.3% of table entries are true singletons stored at count 2, against 0.5% at 0.1%
FPR, which costs 2.0 GB instead of 1.3 GB for the month. Use 0.1% for tables that will ship.

**Position books: behaviour, routes, personal books, over the board** (`doppel_lab/posbook_*.py`,
`lab/results/posbook_explore.json`; chart-ready series under its `charts` key). Counts are of the *decision
position*, the one the mover faces. Train games are scored against the train book minus their own visits.

*Popularity predicts behaviour, and the position key (R0) predicts it better than the move-order key (R1)*
(`posbook_behave.py`). At plies 8–20 in the 1500 band (100k train games, 5.2 M moves), think time falls from
4.4 s in a position no other train game reached to 1.8 s in one reached by 65k–262k games. The blunder rate falls
from 6.7% to 0.5%. Maia-3 top-1 stays flat at 48–54%. Controlling for ply × rating × clock × time control
cells, the position count's share of within-cell variance (partial R²) is:

| Target (1500 band, train) | R0 alone | R1 alone | Both | R0 effect, count 0 vs ≥ 262k (both in) |
|---|---|---|---|---|
| Think, logit share of clock | 0.104% | 0.052% | 0.131% | +0.23 logit (about +26% time) |
| Blunder (wc loss ≥ 0.3) | 0.037% | 0.006% | 0.048% | +5.3 pt |
| Maia-3 top-1 | 0.003% | 0.016% | 0.074% | +0.24 (R1: −0.27) |
| Maia-3 log-prob, played move | 0.006% | 0.061% | 0.131% | +0.59 nats (R1: −0.87) |

Per-move behaviour is noisy, so all the R² are small. The comparison is what counts. For think time and
blunders, R0 explains 2× and 6× what R1 does. Adding R1 to R0 raises the R² by a quarter; adding R0 to R1 more
than doubles it. The heldout eval games give the same ordering in every band (`behaviour_bands`). For think
time, R0's partial R² is 1.4× R1's at 800 and 1.5–2.3× from 1200 to 2400. It grows with rating, from 0.06% at 800 to 0.59% at 2400. Strong players' think time tracks
popularity more: at plies 8–20 the think share of the clock falls 4.4-fold from rare to common positions at
2400, against 2.3-fold at 800.

Maia-3 is the exception, and the reason is mechanical. The teacher reads an 8-position history window
(`teacher.py`), so its accuracy depends on the route. At *transposed* plies (common position R0 ≥ 16, rare route
R1 ≤ 1; 116k train plies, same cells), Maia-3 top-1 is 49.8%, against 53.3% in book by both keys and 52.8% out of
book. People behave the other way there: they think 6% less than in book (logit −0.06), and blunder 3.7%
against 5.2%. Humans track the position; Maia-3 tracks the path.

The exit event study confirms this (train 1500, exits at plies 6–40). Ply −1 is the move into the first unseen
position. When the two keys exit together (26k games), think time jumps from 3.5 s to 5.1 s at that move. When
the position exit comes at least 4 plies after the move-order exit (35k transposed games), the move-order exit
brings a small bump, 2.5 s → 3.3 s, and then think time drops back to 3.0 s. At the later position exit it
jumps 3.7 s → 5.3 s and stays near 4.8 s. **A Doppel should key its book features, and the think-time head's
book input, by position (R0).** For the weird-move rate the two keys give the same answer (table above).

Self-agreement (`noise_floor.py`'s method, R0 key, heldout 3+0, plies 6–29, 3.3 M repeated (player, position)
pairs) rises with popularity. When the same player reaches the same position again, they repeat their move 70.5%
of the time in positions with < 2 train games and 87.1% in positions with ≥ 262k. Two different players at the
same positions agree 31–42%, and that rate falls slightly with popularity, because popular positions are branch
points. The personal component of the noise floor is largest where theory is densest.

*Transpositions* (`posbook_transpose.py`, heldout 2026-10, 12.7 M games, plies < 40). In 1.5 M games, 12.7% of
visits to the 2,617 positions with ≥ 500 visits arrive by a non-modal move order. That share is 4% at ply 3,
29% at ply 6, and 49% at ply 12. The most-transposed positions are main-line hubs:
- Four Knights (ply 6, 35 routes, 53% non-modal; 3.Nc3 Nf6 47%, 2…Nf6 3.Nc3 Nc6 23%, 2.Nc3 14%).
- QGD Normal Defense (ply 6, 53%; 1.d4 Nf6 2.c4 e6 3.Nc3 d5 17%).
- Zukertort 1.d4 d5 2.Nf3 (16% via 1.Nf3 d5 2.d4).
- The Slav (28% via 1.d4 c6).

The positions with ≥ 1k visits and the highest non-modal share (88–94%) are all London System and Semi-Slav or
Queen's Gambit structures. Each has 41–276 routes, none above 12%. These are system openings, where move order
barely matters.

The R1 residual (common position, rare prefix) is 3.6% of common-position plies; 16.5% of games have one. By
opening family (≥ 20k games), it is highest in the systems and the flank openings:
- Van Geet 9.1%, Saragossa 8.7%, Semi-Slav 8.5%, London 7.8%, King's Indian 7.5%, English 6.6%, Zukertort 6.6%.
- Lowest in forcing gambits: Scotch 1.3%, Englund 1.1%, Danish 0.8%, Elephant 0.6%, Grob 0.3%.

By ECO letter: A 5.2%, D 4.9%, E 7.0%, against B 2.9% and C 2.6%. The route a player takes is a style. Over the
156k players with ≥ 40 heldout games, a player's residual rate has split-half r = 0.55 (Spearman-Brown 0.71).
After regressing out rating, r = 0.50 (0.67). The median rate rises from 1.9% at 800 to 4.3% at 2000–2200.

*Personal books* (`posbook_personal.py`). Players are reid_k.py's: 8,000 heldout players with ≥ 80 3+0
games. Targets are each player's last 40 games, and the history is their first k. Band books are the
2026-10 train games of the player's 200-point band (`data/v0/posbook/band/`, R0, count ≥ 2).

Coverage is shallow:
- After 40 games, 10.7% of a later game's own decisions are in the personal book: 38.5% at plies 1–10 and 1.1% at
  11–20.
- The band book covers 33.9%: 93.6% at plies 1–10 and 31.6% at 11–20.
- Personal-book exit is ply 5.3, against 14.2 for the band book. Almost nothing personal lies outside the band
  book (0.3% of games).

Prediction is where personal books pay off. At personal-book positions (150k sampled decisions, k = 40), the
player's own earlier modal move is the move played 82.2% of the time. The band book's most common move from that
position is right 46.8% of the time. Own right and band wrong happens 40.2% of the time; the reverse, 4.8%. With
3+ earlier visits, own-mode top-1 is 88.6%. In rare positions (< 100 band games) it is still 77.3%, against 43.9%
for the band book.

As a fingerprint, the personal book beats every surface feature. Each player is a bag of the positions they
created with their own moves (plies < 30), weighted by rarity in their band book (log N / (count + 1)). Cosine
between a k-game probe and the 40-game gallery, in pools of 1,000 (chance 0.1%), gives:

| Probe games k | 1 | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| Surface features (reid_k.py) top-1 | 0.9% | 2.9% | 9.4% | 17.6% | 23.6% | 33.0% |
| Personal book, rarity-weighted | 11.2% | 18.5% | 33.6% | 50.4% | 66.2% | 80.2% |
| Book + surface (sum of z-scores) | 12.1% | 20.2% | 37.7% | 56.6% | 72.2% | 84.6% |

Restricting the tokens to divergent moves (< 1% of band games from a position with ≥ 100) is much weaker: 7.2% at
k = 10. Half of 10-game probes contain no such move. Identity lives in the repertoire as a whole, not in the odd
moves. For the multi-game z encoder (the v0e successor), opening repertoire carries far more identity than
surface style (17.6% → 50.4% at k = 10 from openings alone). A position-keyed personal book is a cheap R2
baseline to beat.

*Online vs over the board* (`posbook_otb.py`; Lumbra's Gigabase OTB, research only under CC BY-NC: aggregates
here, every table under `data/raw/lumbras/`). The 13.25 M OTB games were converted by `poskey pgn` in about 2 min.
Lumbra's merges sources, and the same game recurs: before dedup, 11% of OTB games of ≥ 60 plies shared their
ply-60 position with another game, against 0.01% on Lichess. Dropping exact repeats (≥ 40 plies, equal first 80
plies; 605k games) leaves 12.65 M. After that, 1.5% still share the ply-60 position, 6–8% before 1990, from
near-duplicates and deep theory. Pre-1990 depths are upper bounds.

The probe is 253k OTB games (a 2% hash sample, not in the book) plus the Lichess eval games. At equal book size
(12.4 M games, R0), the mean book exit in plies is:

| Games | in the Lichess book | in the OTB book |
|---|---|---|
| Lichess 2026-10 blitz | 14.4 (median 14) | 12.9 (12) |
| OTB | 17.7 (17) | 21.9 (21) |

OTB games stay in book longer even in the online book. They also gain more per doubling of
their own book: 1.02 plies, against 0.71 for Lichess games in theirs. By ply 16 (full books), 74% of OTB positions
occur in the Lichess book, while only 33% of Lichess positions occur in the OTB book (48% in their own).

By rating band (200 points; FIDE Elo for OTB, Lichess blitz rating for Lichess, which are different scales), each
medium in its own band book at equal N:

| Band | 1400 | 1600 | 1800 | 2000 | 2200 | 2400 |
|---|---|---|---|---|---|---|
| Lichess games, Lichess band book | 10.9 | 12.1 | 13.3 | 14.7 | 14.7 | 13.4 |
| OTB games, OTB band book | 14.6 | 16.3 | 18.3 | 20.3 | 19.9 | 18.4 |
| OTB games, Lichess band book | 12.4 | 13.9 | 15.5 | 17.2 | 17.1 | 15.9 |
| Lichess games, OTB band book | 9.7 | 10.6 | 11.6 | 12.9 | 13.2 | 12.0 |

Equal N is the smaller book of the two (0.5 M games at 1400, 0.13 M at 2400). Over the board, players at every
level stay 3.7–5.5 plies deeper in book. Positions that are common in one medium are often rare in the other.
- Online only (in 0.2–0.5% of Lichess games and 0.002–0.01% of OTB games): the Englund and Hartlaub-Charlick
  gambits (1.d4 e5 2.dxe5 d6), 1.e4 d5 2.Nf3 (Tennison), 1.e4 e5 2.Qf3 and 2.Qh5 Nc6 3.Bc4, the McConnell
  2…Qf6, 3…Qe6+ in the Scandinavian, and 1.e4 c6 2.Bc4.
- OTB only (0.2–0.6% of OTB games, almost never online): main-line theory 12–21 plies deep, such as the Closed
  Ruy Lopez to 9.h3 Na5 10.Bc2 c5 11.d4, the Richter-Rauzer to 8.O-O-O, the Semi-Slav Meran and 6.Qc2, and the
  Queen's Indian 4…Ba6.

By decade (OTB own-decade book at 200k games), OTB depth is 20.9 / 19.6 plies in the 1970s / 1980s (upper
bounds, mostly elite games) and 15.9–16.2 from the 1990s to the 2020s. Lichess 2026 games stay in each decade's
OTB book a little longer the more recent it is: 8.1 plies (1970s) → 8.9 (1990s) → 9.3 (2020s). Online openings
are closest to current OTB practice.

*Depth vs corpus size* (`posbook_months.py`; probe: the 2026-10 heldout eval games). The base book is the
2026-10 train games. It grows by the 2026-10 heldout games (minus every eval game), then by the head of
September 2026. That head is the month's first 30 M games, streamed from the Lichess database through `poskey pgn
--lichess-blitz` with nothing raw on disk. Of those, 12.5 M pass corpus.py's filter and 8.0 M remain after
dropping games with a held-out player. The full download
ran at 1–4 MB/s, so the rest of the month was skipped. R1 counts come from `poskey count --key prefix`, which
reproduces book.py's exit exactly (12.013 plies at 22.7 M).

| Book (games) | 22.7 M | 35.2 M | 43.2 M |
|---|---|---|---|
| R0 mean book exit (800 / 1500 / 2400) | 15.00 (11.4 / 14.7 / 17.7) | 15.49 | 15.70 (11.8 / 15.4 / 18.7) |
| R1 mean book exit | 12.01 | 12.40 | 12.55 |

Depth does not saturate. Over the largest book's nested subsamples, 169k to 43 M games, R0 gains 0.70–0.72 plies
per doubling in every range and R1 0.50–0.51. Across the three full books the gain is 0.76 and 0.58. That is
slightly higher, because the heldout increment holds the probe players' own other games. At this rate, one more
ply of R0 book costs about 2.6× the games.

*Client lookups.* dartchess 0.14 (the client's chess library) has `Position.zobristHash()`, ported from shakmaty
with Polyglot tables. It matches python-chess and poskey only with `mode: EnPassantMode.pseudoLegal`; the default
is `legal`, which differs when the en passant capture is pinned. The hash is a signed Dart `int` holding the
unsigned bit pattern, so a shipped table must be sorted and searched by unsigned order (or stored in signed
order). dartchess does not support the web, so lookups work on native builds only.

Band-specific books (`doppel_lab/bandbook.py`, `lab/results/bandbook.json`): six rating groups, each with its own
book of 350k train games, scoring 10k held-out games of every group. Every group stays in book longest, and
plays the fewest weird moves, against its own group's book. The diagonal wins every row. Mean T1 exit, own
book vs the extremes:

| Games from | ≤900 book | own book | 2300+ book |
|---|---|---|---|
| ≤900 | 8.9 | 8.9 | 6.8 |
| 1500 | 9.2 | 10.0 | 8.3 |
| 2300+ | 8.4 | 13.0 | 13.0 |

So two effects are separate. Opening lines are band-specific: a 2300+ game leaves a ≤900 book at ply 8.4.
And depth in one's own band's book still rises with rating (8.9 → 13.0 at equal book size), so strong players
also conform more. The asymmetry is in the weird-move rate: strong players' moves are rarely weird to a weak
book (2.5%), while weak players' moves often are to a strong book (13.5%). For a Doppel, the band-matched book
is the right reference for "weird", not a pooled one.

**Upsets** (gap ≥ 300, 38,930 heldout games). The underdog wins about 21% at every gap from 300 to 700,
against an Elo expectation of 12% → 1.4%, so large Lichess gaps mostly mean a mis-rated account. From the
first 30 plies, upsets are predictable: AUC 0.60 (gap and time control) → 0.78 (+ early T0 features).
Following the held-out sides through the month (`doppel_lab/misrating.py`): big-gap games bunch in an account's
first 10 games of the month (32–50% of these sides, against 11% of all sides). A losing favourite drops 107 Elo
in that one game, a sign of a highly uncertain (provisional or returning) rating. After the game there is no
further drift in any group (−6 to +5). Using the favourite's previous-game rating swing as a pre-game
uncertainty proxy: favourites with a swing of 80+ lose 38% (Elo expects 8.5%), but settled favourites (swing
under 10, 93% of sides) still lose 17.9% against 7.2%. So rating uncertainty explains part of the excess, and
the rest says the Elo curve's tails are too thin for Lichess blitz at gaps of 300 or more.

The Elo curve itself (`doppel_lab/elo_curve.py`, all 34.9 M games of the 2026-10 corpus, `lab/results/elo_curve.json`)
fits strong blitz and fails club blitz. Fitting E = 1 / (1 + 10^(−gap/s)) per mean-rating level gives s = 443
at 2400, 513 at 2000, 688 at 1600 and 759 at 1200, against Elo's 400. Two artifacts sit on top of that:

1. **The 1500 start.** New Lichess accounts start at 1500. A favourite rated 1500–1599 (gap 150–450) scores
   0.483, below a coin flip, against an Elo expectation of 0.796, and that bin holds 2.5× the games of its
   neighbours. Dropping games with either side at 1500–1599 removes the non-monotone dip at the 1200 level.
2. **Wide pairings are selected.** Past a gap of about 300 the curve stops rising (gap 312: 0.811, gap 512:
   0.776, gap 712: 0.812, all levels without 1500s), so a pairing that wide is itself evidence that a rating is
   wrong.

For a Doppel, this means the rating inputs (Maia-3's Elo conditioning, our `elo_self`/`elo_opp`) are much noisier
below 2000 than their face value. Ratings in 1500–1599 in particular carry little information.

Time scrambles (`doppel_lab/scramble.py`, same corpus without 1500–1599 ratings): when both sides fall under
10 s (1.4 M games, 76% of them ending on time, 9% drawn), the fitted Elo scale is 1,152, against 607 in calm
games where neither side drops under 30 s. A favourite by about 175 points scores 0.601 in a scramble and 0.675
in a calm game. Rating matters about half as much once both clocks run down. Across the corpus's blitz time controls
(`doppel_lab/tc_scale.py`), the fitted scale barely moves: 634 at 3+0, 643 at 3+2, 692 at 5+0, 653 at 5+3, while
flag rates fall from 27% to 18%. Lichess pools blitz into one rating, so this is the same rating predicting at
different speeds. A clock change of this size matters far less than whether the clocks actually run low. Selection inflates this somewhat:
a favourite who reaches a scramble has already failed to win cleanly.

**Era** (`doppel_lab/vintage.py`, the first 150k games of each March 2013–2026, `lab/results/vintage_lichess.json`).
Books are rebuilt per year at equal N, tiers are within-year rating percentiles, blitz only.

| | 2013 | 2017 | 2021 | 2026 |
|---|---|---|---|---|
| Own-era book exit, T1 (ply) | 9.45 | 9.36 | 9.29 | 9.24 |
| Weird-move rate | 2.5% | 2.5% | 2.7% | 2.5% |
| Exit from today's book, T10 (ply) | 10.36 | 10.78 | 10.75 | 10.78 |
| Flag rate | 28.3% | 29.1% | 23.9% | 24.5% |
| Mean think per move (s, clocks from 2018) | — | — | 4.69 | 4.51 |
| Rating, p25 tier / p95 tier median | 1396 / 1990 | 1324 / 2170 | 1154 / 2276 | 1172 / 2374 |

Online blitz is close to stationary on surface features. The tier gradient (bottom quarter → top 5%: book exit
8.5 → 10.5, weird rate 3.8% → 1.6%) is about ten times any change across years, and holds every year. The
rating scale widened by about 600 points between the p25 and p95 tiers, which is why tiers are percentiles. Sampling check: the first 100k games of a month (2026-10, held-out stream) match a random 100k of the same
month within 1.4 rating points, 0.5 plies and 0.3 pt of 1.e4 share; the flag rate is 1.0 pt higher on day 1. So the
month heads stand in for the month, and the flag-rate decline (about 4 pt) is larger than the sampling bias.

**Openings and results** (`doppel_lab/opening_perf.py`, 2026-10 corpus without 1500–1599 ratings): White
scores above the Elo expectation in every band, least at 1600–1800 (+1.7 pt) and most at 2400 (+2.7 pt). Draws
rise from 3.4% at 1200 to 8.1% at 2400. Per 4-ply line (≥ 20k games in a rating group), the residual against
the group's mean is ±2–5 pt. At 1100–1499, 1.e4 e5 2.Nf3 f6 is the best line for White (+4.4 pt; Black's
Damiano loses) and 1.e4 d5 2.Nf3 dxe4 the best for Black (−4.7 pt). At 1600–1999, 1.e4 e5 2.Nf3 d5 (the
Elephant Gambit) scores −4.1 pt for White. At 2000+, both Scandinavian main lines favour Black (−2.1 and
−2.2 pt). These residuals mix the opening with the kind of player who chooses it.

Leaving the book costs a little (`doppel_lab/novelty.py`, 10.6 M held-out blitz games; book = the 22.7 M-game
train table). The side that plays the first move into a position with fewer than 2 book games scores about
1 pt below its Elo expectation, after removing White's edge. When that move comes from a well-known position
(100+ book games), the side found an almost unplayed move in a common position, and the cost rises to 3–4 pt:
−4.4 pt at 800, −3.0 pt at 1600–2000, −3.9 pt at 2400. The later it happens, the worse: −3.3 pt before ply 8,
−6.7 pt at plies 14–19. An off-book move in a known position is usually a mistake, at every level. That
supports a "weird move" feature that conditions on how well known the parent position is.

**Opening fashion** (`doppel_lab/fashion.py`, `lab/results/fashion.json`): shares of named first-move lines by
year, online (Lichess March heads) and over the board (TWIC classical, open events).

| Line | Online 2013 | Online 2026 | OTB 2012 | OTB 2026 | OTB 2600+ 2012 | OTB 2600+ 2025 |
|---|---|---|---|---|---|---|
| London | 2.1% | 6.2% | 1.4% | 2.9% (peak 4.1% in 2018) | 0.2% | 1.7% |
| Caro-Kann | 2.6% | 5.2% | 3.8% | 5.9% | 4.4% | 2.6% |
| Sicilian | 9.8% | 9.6% (peak 12.0% in 2020) | 20.4% | 22.1% | 13.3% | 14.9% |
| Italian | 3.1% | 3.5% | 1.5% | 3.2% | 1.2% | 9.6% |
| Queen's Gambit | 4.7% | 3.5% | 7.1% | 6.4% | 12.1% | 7.3% |

Fashion moves by several points over a decade, unlike the surface features above, and it moves differently by
medium and tier. The London grew online and stayed, while over the board it peaked in 2018 and faded. The
elite's Italian renaissance barely shows below 2600. A natural experiment came out null: after Netflix's "The
Queen's Gambit" (October 2020), the Queen's Gambit share online fell from 2020-03 to 2021-03 (4.2% → 3.9%,
bottom quarter z = −5.3). Only the top 5% rose (2.4% → 2.9%, z = 2.3).

**Populations** (`doppel_lab/populations.py`, rating-matched by coarsened exact matching on year, time control,
own and opponent rating). Arena tournament vs lobby games, a competitive-vs-casual proxy, differ very little:
+1.2 plies, 0.04 s faster per move, 0.3 pt fewer weird moves. At matched rating, titled players (blitz, 6,670
sides) score 2.0 pt above their Elo expectation (z = 3.2) and play 0.5 pt fewer weird moves (z = −3.1).

**Age at the board** (`doppel_lab/age.py`, `lab/results/age.json`): TWIC classical over-the-board games
2012–2026 joined to birth years from the FIDE rating list (3.4 M sides). Aggregates only; no per-player output.
Each age bin against players aged 25–34 at the same FIDE rating, opponent rating and year:

| Age | Score − Elo expectation | Book exit (plies) | Game length (plies) | Draw rate |
|---|---|---|---|---|
| 6–11 | +4.3 pt | +1.7 | +6.2 | +3.4 pt |
| 12–14 | +4.7 pt | +1.5 | +4.4 | +2.6 pt |
| 15–17 | +3.3 pt | +1.1 | +2.2 | +1.4 pt |
| 18–24 | +2.0 pt | +0.5 | +1.3 | 0.0 |
| 35–49 | −2.0 pt | −0.4 | −2.7 | +1.8 pt |
| 50–64 | −3.5 pt | −0.9 | −4.6 | +3.0 pt |
| 65+ | −4.7 pt | −1.3 | −5.4 | +4.0 pt |

All |z| > 10 except the 18–24 draw rate. The score column is the known rating lag: a junior's rating trails
their improvement, an older player's trails their decline. The other columns are style at equal strength:
younger players go deeper into book and play longer games. Draw rates are U-shaped in age.

**Over the board** (`doppel_lab/twic_science.py`, TWIC issues 920–1669, 2012–2026, 2.9 M games,
`lab/results/twic_science.json`). Medium, speed and age group are classified from TWIC's own conventions: a
Site ending in " INT" marks online play, and event names mark speed and junior/senior events. Era, classical
open events, own-year books at equal N (25k games):

| | 2014 | 2018 | 2021 | 2026 |
|---|---|---|---|---|
| Own-year book exit, T1 (ply) | 12.72 | 12.51 | 13.27 | 13.19 |
| 6-ply opening entropy (bits) | 9.19 | 9.24 | 8.90 | 8.76 |
| 1.e4 / 1.d4 | 45% / 36% | 45% / 35% | 47% / 35% | 50% / 31% |
| Draw rate | 31.7% | 31.7% | 33.2% | 33.4% |

Unlike online blitz, over-the-board classical play changed, with a step at the 2021 return from the pandemic:
deeper book, fewer distinct openings, more 1.e4, slightly more draws. Draw rate rises steeply with strength in
every year (about 25% below 2200, 55% at 2600+).

Matched contrasts (FIDE rating in 50-point bins and year):

| Contrast | Draws | Short draws (≤ 40 plies) | Book exit | Weird moves | Plies |
|---|---|---|---|---|---|
| Online vs OTB, fast games | −9.8 pt (of 19.0%) | −1.8 pt | −0.7 | +1.3 pt | +0.6 |
| Junior vs open, classical OTB | −0.3 pt | −2.8 pt (of 4.1%) | +1.8 | −1.0 pt | +5.9 |
| Fast vs classical, OTB | −17.8 pt (of 39.9%) | −4.1 pt | −1.8 | +0.7 pt | +4.9 |

Titles at matched rating (per side, classical OTB): no strength premium. Score minus Elo expectation is −0.5 pt
for GMs (z = −0.8) and +0.3 pt for IMs. GMs draw 5.2 pt more than untitled players of the same rating and
leave book 0.5 ply earlier; CMs, often juniors, score 1.7 pt above expectation (z = 13), consistent with the
age table's rating lag.

Short draws over the board (`doppel_lab/short_draws.py`, TWIC classical; ½–½ in 20 moves or fewer): 1.5% of
games at 1400, 5.1% at 2200, peaking at 9.0% at 2500, then 2.9% at 2700, where elite round-robins discourage
them. Evenly matched pairs agree most: 10.1% within 100 points, 0.7% at 400+. Last rounds double the rate
(pairs within 200 points): at 2200–2499, 14.6% in the last round vs 8.4% in middle rounds, and 13.8% vs 8.7%
at 2500+. That is tournament strategy, not chess skill, and it is something an OTB Doppel would need to know
the event context to copy.

**Tilt** (`doppel_lab/tilt.py`, `lab/results/tilt.json`): 2.7 M pairs of consecutive games by the same
player within 60 minutes, Lichess month heads 2018–2026 (562k sessions). Effects are within session (fixed
effects) minus a permutation null that shuffles game order within each session. The naive within-player
estimate is badly biased by overlap (a game's score is one pair's outcome and the next pair's regressor): raw
+12.7 pt "reverse tilt" in score, of which the null reproduces +12.9 pt (Miller & Sanjurjo's finite-sequence
bias). Corrected, after a loss compared with after a win:

| Next game | After a loss | After two losses |
|---|---|---|
| Score − Elo expectation | −0.28 pt (z = −4.5) | −0.67 pt (z = −7.5) |
| Lost within 40 plies | +1.4 pt (z = 38) | +1.1 pt |
| Think per move | −0.12 s (z = −51); a further −0.17 s if the loss was on time | −0.07 s |
| Same time control as before | −0.8 pt | −0.9 pt |
| Plays again within 60 min | −4.0 pt (83.8% vs 87.7%, z = −87) | |

Tilt is real but small in results, and clear in tempo: players speed up and lose quickly more often after a
loss, more so after losing on time. For a Doppel, the previous game's result is a cheap conditioning signal
for the think-time head.

**Adapting to the opponent** (`doppel_lab/opp_adapt.py`, 4.5 M held-out 3+0 sides, player x colour fixed
effects). Players barely change their opening for the opponent. They leave their modal first move or reply
21.6% of the time, and that moves by under 1 pt from a 300-point-weaker to a 300-point-stronger opponent. Pace
does move. Against opponents 50–150 points stronger, players take 0.04–0.10 s more per move, and 0.12 s more
at +300. Against weaker ones they take 0.04–0.09 s less. Part of this is the game itself, since stronger
opponents set harder positions. A Doppel's opening choice can ignore the opponent's rating; its think-time
head can use it (`elo_opp` is already an input).

**Session fatigue** (`doppel_lab/fatigue.py`, 3.3 M sides in 563k sessions, session fixed effects, game
index dummies vs game 1). The score effect is not identified. With all games, scores fall 1–2.6 pt later in
a session. Dropping each session's final game flips that to +1.2–2.5 pt. Both come from the stopping rule:
players quit after losses, so final games skew toward losses and kept games away from them. Two effects hold
in both versions. Players move faster as a session runs on, by 0.05–0.14 s per move (all games) or 0.10–0.30 s
(final game dropped) at games 5–17+. They also lose quickly less often (early loss −0.6 to −0.9 pt). A
session-index input to the think-time head is cheap; a "fatigue" score penalty has no support here.

Rematches (`doppel_lab/rematch.py`, 230k pairs of games between the same two accounts within 60 minutes):
the first game's winner scores 0.554 in the rematch. Against the expectation from the rematch's own ratings,
which already include game 1's rating exchange, that is −0.2 pt (SE 0.1). The split by colour is White's
edge, +1.5 pt as White and −1.8 pt as Black. There is no matchup edge and no loser's tilt beyond what the
ratings already price in.

**Style persistence** (`doppel_lab/persistence.py`, `lab/results/persistence.json`): the bar for a style
latent. 20,000 held-out players with at least 40 games at 3+0 in 2026-10 (one time control, so time-control
choice can't pass for style); each player's games split into odd and even halves. Split-half Spearman across
players after regressing out rating: mean think 0.93, fast-move share 0.93, think CV 0.88, book exit 0.88,
time-trouble share 0.80, weird-move rate 0.79, flag losses 0.78, resignations 0.62. Opening choice is the most
stable of all (1.e4 0.97, 1.d4 0.97, ...c5 0.97, ...e5 0.97 against 1.e4). Re-identification (match a player's
half A to their half B among 1,000 players): style alone top-1 64%, rating alone 45%, style and rating 81%
(top-10 97%); chance 0.1%. A z that claims to capture the player should at least re-identify this well. The human noise floor
(`doppel_lab/noise_floor.py`, 5.1 M held-out 3+0 games; a position is the move-order prefix): when a player
reaches the same position again, they repeat their move 84% of the time at ply 4, 81% at ply 10–14 and 77% at
ply 20. Two different players in the same positions agree 32–58% of the time. Predicting each visit from the
player's own other visits gets 79–85% top-1. So in repeated positions, which are mostly openings, a perfectly
personalised model tops out near 80% top-1, against about 45% for a perfect population model. That gap is the
room a per-player z has to fill. Consistency rises with strength (plies 8–16 pooled): self-agreement is 0.75
at 800, 0.80 at 1600, 0.86 at 2000 and 0.92 at 2400. Strong players have a prepared, fixed repertoire, so a
strong player's Doppel can predict them better. Later, unrepeated positions have no direct floor. How fast the signal accumulates
(`doppel_lab/games_needed.py`, 8,000 of those players with 40+ games): rating predicted from the same style
features averaged over a player's first k games in time order, never from rating itself, goes from R² 0.18
(MAE 265) at one game to 0.42 (224) at 5, 0.52 (203) at 10, 0.59 (186) at 20 and 0.66 (169) at 40, against MAE
298 for a constant. That is about +0.08 R² per doubling. It is the data-needs curve a per-player z should beat:
the same accuracy from fewer games. Re-identification by k (`doppel_lab/reid_k.py`, 8,000 players with 80+
games): the first k games are matched against a gallery of each player's last 40, in pools of 1,000. Top-1 is
1.1% at k = 1, 10% at 5, 17% at 10, 25% at 20 and 33% at 40 (top-10: 8%, 36%, 51%, 62%, 71%). Drift is small
(`reid_k.py drift`, players with 160+ games, 10-game probes). Probes right before the gallery get 21.2% top-1;
probes 110 games earlier get 18.7%. The persistence study's 64% comes from more evidence: each half there is
half of a player's games, often well over 40. Style is close to stationary over a month, and evidence size
dominates. A multi-game z should see at least 10–20 games.

Style transfer across time controls (`doppel_lab/style_transfer.py`, 2,277 held-out players with 20+ games at
both 3+0 and 5+0): the cross-time-control Spearman of each feature against its within-3+0 split-half
reliability. Opening choice transfers at 0.88–0.97 of its reliability, book exit 1.03, weird-move rate 1.06,
think CV 1.01, time-trouble share 1.02, flag losses 0.93. Mean think keeps its rank order (0.84) but rescales
(3.9 → 5.2 s per move, 2.2% → 1.7% of the base clock). So one z per player can serve several blitz speeds, with
the think-time head scaling by the clock.

Repertoire breadth (`doppel_lab/repertoire.py`, 21k held-out players, entropy of their first two moves as White
over exactly 30 games at 3+0) is U-shaped in rating: 1.76 bits at 800 (unsettled), 1.25 around 1500–1600 (a
fixed repertoire), 1.53 at 2400 (deliberate variety); Black replies to 1.e4/1.d4 follow the same shape (1.43 →
1.23 → 1.47). Within a rating bin, a narrower repertoire goes with slightly better results than the rating
predicts (Spearman −0.02 to −0.14, strongest at 800–900), and the link is gone above 2200.

**Encoder benchmark** (`doppel_lab/bench.py`, `bench_embed.py`, `bench_doppel.py`). This is the harness for the
question above: can a learned match encoder beat surface features? Every model sees only the game's move and
clock stream, never headers, and all are scored on the same games and folds (grouped by White's account). The
Lichess suite has 91.5k games from 2018–2026 March heads. Maia-3 gets 1500 vs 1500 as its rating input so it
cannot read the label. Surface features: T0 side aggregates, book exits and weird moves against the 2026-10
book, and per-side likelihood under each band's book (`bandbook.py`). Gradient-boosted trees, 5-fold CV:

| Task | Surface, v1 | Surface + band books |
|---|---|---|
| Mean rating (R²) | 0.406 | 0.492 (MAE 199) |
| Within-year rating percentile (R²) | 0.395 | 0.487 |
| Top 5% of the year (AUC) | 0.832 | 0.885 |
| Arena vs lobby (AUC) | 0.701 | 0.703 |
| Titled player, among the top 10% (AUC) | 0.666 | 0.689 |
| Year, 2023+ vs ≤ 2020 (AUC) | 0.581 | 0.582 |
| Rapid vs blitz (sanity, AUC) | 0.999 | 0.999 |

A second suite, TWIC (80k games 2012–2026, FIDE ratings, no clocks, so move-only), is much harder for surface
features: FIDE rating R² 0.094 (MAE 180.5 against 187.5 for a constant), year R² 0.04, era 2021+ AUC 0.62,
online vs OTB among fast games 0.63, junior 0.56, fast vs classical OTB 0.56 (`lab/results/bench_surface_twic.json`).
This is where an encoder has the most room.

The arena score is mostly a clock artifact: its top permutation features are end-of-game clock, median think
and tempo jitter, because arenas use a different time-control mix (84% without increment, against 73% in lobby
games) and berserk halves the clock. The matched contrast above, with time control matched and berserk games
dropped, finds almost no behavioural difference. Read arena AUC as a time-control detector.

Maia-3 79M, rating-blind, on the Lichess suite (`lab/results/bench_maia3.json`). maia_linear is the mean
final-layer embedding per side (PCA 256 + ridge/logistic); maia_policy is trees on the policy scalars (mean
played-move log-prob and its spread, entropy, top-1 match, mean rank):

| Task | Surface + books | Maia linear | Maia policy | Surface + Maia policy |
|---|---|---|---|---|
| Mean rating (R²) | 0.492 | **0.532** | 0.320 | **0.564** |
| Rating percentile (R²) | 0.487 | 0.529 | 0.303 | 0.553 |
| Top 5% (AUC) | 0.885 | 0.880 | 0.670 | 0.889 |
| Arena (AUC) | 0.703 | 0.610 | 0.534 | 0.705 |
| Titled (AUC) | 0.689 | 0.627 | 0.549 | 0.696 |
| Era 2023+ (AUC) | 0.582 | 0.548 | 0.511 | 0.584 |
| Rapid (sanity, AUC) | 0.999 | 0.651 | 0.573 | 0.999 |

A frozen general-purpose encoder beats the surface features on rating from moves alone (R² 0.532 vs 0.492),
even though it never sees the clocks the surface features use, and the two combine to 0.564. On everything
clock-driven (arena, rapid) and on titled players, the surface wins. Era is close to chance for every view.
These are the bars the Doppel z (`bench_doppel_<run>.json`) must clear.

**Maia's rating sweep is a much better rating estimator than its embedding** (`bench_embed.py probe`, sweep
features = per-side mean log-likelihood of the moves played at 9 rating inputs 800–2400, plus the argmax
rating). From one game, rating R² 0.744 (vs 0.532 embedding, 0.492 surface), percentile 0.754, top 5% AUC
0.959, titled 0.744. Surface + policy + sweep: 0.783 rating R², 0.966 top 5%. The sweep adds nothing on era or
clock tasks (arena 0.631, rapid 0.631). This is the real bar for any Doppel z on rating: 0.744 from one game,
from a model that was trained with ratings as an input.

On TWIC (move-only, `lab/results/bench_maia3_twic.json`), Maia's embedding wins every task. FIDE rating R²
rises from 0.094 to 0.140 and year R² from 0.036 to 0.064. Era 2021+ AUC goes from 0.619 to 0.688, online vs
OTB among fast games from 0.630 to 0.703, fast vs classical OTB from 0.564 to 0.652, and junior from 0.563 to
0.581. Without clocks the surface has little to work with, and move quality carries the signal. Online play
and fast OTB play are visible in the moves themselves.

The rating sweep on TWIC (`lab/results/bench_maia3_twic.json`) doubles FIDE rating R² from 0.140 (embedding)
to 0.282, 0.306 with surface and policy. It is far below the Lichess 0.744 because TWIC ratings sit mostly
above 2000, where the sweep's top input (2400) and Maia-3's training range run out, and the rating range is
narrow. On era, online vs OTB, junior and fast OTB the sweep adds little over the embedding.

The v0d Doppel z (`doppel_lab/bench_doppel.py`, `lab/results/bench_doppel_v0d.json`; the posterior mean per
side and game half, 8 dims) is far weaker than both. Rating R² is 0.207 (trees; 0.144 linear) against 0.492 for
the surface and 0.532 for Maia's embedding. Top 5% AUC is 0.714, titled 0.599, arena 0.686, era 0.576. Added
to the surface it changes nothing (rating 0.493, arena 0.715). Its 8 dims hold some rating and clock-style
information, and none of it is new.

**Pooled over a player's games** (`doppel_lab/players_suite.py`, `lab/results/players_pool_v0d.json`; 2,000
held-out 2026-10 players, 80 3+0 games each; each per-game vector averaged over the player's first k games;
rating R² by trees, 5-fold by player; re-ID = first k games vs a gallery of the last 40, top-1 in pools of 500):

| View | R² k=1 | k=5 | k=10 | k=40 | Re-ID k=1 | k=5 | k=10 | k=40 |
|---|---|---|---|---|---|---|---|---|
| Surface | 0.113 | 0.312 | 0.427 | 0.572 | 2.0% | 15.9% | 27.6% | 50.0% |
| v0d z (8 dims) | −0.078 | −0.038 | 0.015 | 0.066 | 0.6% | 0.9% | 1.1% | 2.8% |
| Maia emb (PCA 64) | 0.310 | 0.637 | 0.739 | 0.867 | 7.5% | 27.3% | 46.5% | 77.1% |
| Maia rating sweep (9 dims) | 0.575 | 0.759 | 0.794 | 0.838 | 1.4% | 2.9% | 4.7% | 10.7% |
| Surface + Maia | 0.396 | 0.693 | 0.760 | 0.880 | 9.1% | 41.0% | 63.5% | 90.5% |
| Surface + Maia + sweep | **0.644** | **0.847** | **0.883** | **0.937** | **9.4%** | **42.4%** | **66.8%** | **91.8%** |

(Rerun 2026-11-07 with deterministic folds; rating R² moved by ≤ 0.02 from the first run, re-ID unchanged.)

The rating sweep and the embedding measure different things. The sweep (how likely the moves are at each
rating input) is the best single rating view at every k (0.575 from one game, vs 0.310 for the embedding), but it
barely identifies anyone (10.7% at k = 40): it measures strength, and many players share a strength. The
embedding is the reverse: it carries who the player is. Together they reach R² 0.937 and 91.8% re-ID at k = 40.
So a v0e z should get both: the sweep as a strength feature, and a set encoder over per-game embeddings for style.

A frozen Maia-3 embedding, averaged over 40 games, identifies a player among 500 in 77% of cases (90.5% with
the surface features), and gives rating R² 0.87. Averaging is enough; nothing was trained. The v0d z does not
pool: averaging adds nothing because there is almost no per-player signal in it to begin with (re-ID 2.8% at
k = 40, chance 0.2%). Within one rating band (this suite spans all bands, so rating is a large share of what
Maia's vector identifies), v0d z's rating R² near zero is expected; its failure at re-ID is not.

For v0e this sets the bar and suggests the design: a multi-game set encoder whose input per game is at least
as rich as Maia's embedding should reach ~77% re-ID at k = 40 for free. A z that cannot beat "average Maia's
embedding" is not modelling the player.

**Doppel v0d gate** (`doppel_lab/v0_gate.py`, `lab/results/v0_gate_v0d.json`, step 32,811, 1500 band): fail,
2 of 5 checks pass.

1. Accuracy, pass: top-1 0.5597 with the prior z, against Maia-3's 0.5601.
2. Think time, pass: log-seconds r 0.668 with the prior z (0.709 with the posterior), against ChessMimic's 0.56.
3. z used, fail. 8 active units and a z-swap gain of +0.0015 nats (CI above 0), but the KL is 2.96 nats per
   half-game, under the free-bits-plus-one bar of 4.0. The z carries barely more than the free bits allow.
4. Persistence, fail. Half-game correlation is −0.024 for the Doppel and −0.055 for Maia-3, against +0.039 for
   humans, whose CI is [0.009, 0.072]. The human signal is itself tiny at this resolution, so this check
   has little power.
5. Variance, fail. The KS distance of per-game ACPL to humans is 0.091 for the Doppel and 0.071 for Maia-3.
   Mean ACPL is 68.0 for the Doppel, 67.3 for Maia-3 and 61.8 for humans.

The variance check is within noise (`doppel_lab/acpl_shape.py`, all three sources at 20k nodes, games
adjudicated at ±600 cp held for 6 plies, which is close to where humans resign). The Doppel's and Maia-3's
per-game ACPL distributions are nearly identical. Medians are 63.9 and 61.5, the IQR is 41.1 for both, and 6.6%
vs 7.4% of games are above 120. Both are worse and wider than humans: median 57.9, IQR 37.0, 4.6% above 120.
The gate statistic, KS(Doppel) − KS(Maia-3), has a bootstrap 95% CI of [−0.010, +0.045], which includes 0.
The check should compare CIs, not point values. The real gap, synthetic play worse than human, is shared with
Maia-3. It matches the policy-implied blunder rate in §7: sampled from at 1500, Maia-3 blunders 7.6% of the
time against 6.75% for humans. Temperature fixes the rate (`doppel_lab/temperature.py`, the same 6,000 positions; the
stored labels agree with the fresh per-move flags on 99.1% of played moves, so the gap is not label noise).
The implied blunder rate is 7.60% at T = 1, 6.98% at T = 0.9 and 6.69% at T = 0.85, against 6.60–6.75% for
humans. T ≈ 0.87 matches. The cost: mean top-move probability rises from 0.557 to about 0.60 and entropy falls
from 1.34 to 1.18. Synthetic play becomes slightly too predictable, and likelihood is best at T = 1 by
construction. Matching both needs the residual to move mass off blunders specifically, not a global
temperature. The excess sits in the policy's tail (`temperature.py where`). Where Maia's top move is almost
forced (p_top > 0.9), humans blunder 0.7% and sampled Maia 1.7%. Where a quarter to a half of legal moves
blunder, the rates are 3.0% vs 4.9%. In the hardest positions (> 90% of moves blunder), sampled Maia is close
to humans (23.4% vs 21.3%). The long tail of low-probability moves adds up to blunders humans don't make.
Truncating the tail does no better (`temperature.py truncation`). Reaching the human rate takes min-p 0.05 or
top-p about 0.93. That removes the human's actual move from the support in 4.8% or about 4% of positions, and
cuts entropy to 1.08, below temperature's 1.18 at the same blunder rate. Humans do play tail moves; they just
don't play the tail's blunders. Of the global fixes, T ≈ 0.87 costs least. The precise fix is the residual:
learn to move mass off blundering tail moves specifically, for example with a blunder-aware auxiliary target
from per-move labels like these.

The simplest fix is the rating input (`temperature.py elo`). The same 6,000 positions come from players
averaging 1550. Run at self and opponent rating 1700, Maia-3 blunders at the human rate (6.67%, 1.01x).
Human-move log-likelihood is also best there among the inputs tried: −1.3038 at 1700, against −1.3077 at
1500, −1.3335 at 1900 and −1.4090 at 2100. A quadratic through 1500/1700/1900 puts the likelihood peak near
1620, so the likelihood-optimal offset on these positions is about +75, not +150. The blunder-rate match at
1700 is partly the temperature excess above: a stronger input trims the tail.

**The offset is small on average and depends on time control** (`doppel_lab/maia_offset.py`, the 9-rating
sweep on the 91.5k-game bench suite, per side, best input by quadratic refinement of the pooled log-likelihood,
`lab/results/maia_offset.json`). For 2024–26 the offset is +2 to +55 in every 100-point band from 900 to
2300; at 1500 it is +10. It is not a constant "+150" correction. Two things move it more than rating does:
- **Era.** Sides rated 1450–1649: +7 (2018), +33 (2019), −70 (2020), −116 (2021), −88 (2022), −41 (2023),
  +5 (2024), +2 (2025), +27 (2026). The 2020–23 dip matches the pandemic influx of new players, who play
  worse than their rating says.
- **Time control.** Same band: 3+2 +54, 5+3 +97, 3+0 −45, 5+0 +7, 10+0 −111, 10+5 −65. With an increment,
  players at a given rating play like stronger players; at 10+0 like weaker ones. Maia-3's input carries the
  rating but not the time control, so its rating conditioning absorbs the difference.

At the extremes (2400+) the offset turns strongly negative in every era (−40 to −280), which is Maia-3's
training range running out. For v0's teacher: no global offset. A per-time-control one is barely worth it
(`doppel_lab/maia_offset_gain.py`, offsets fit on half the sides and scored on the other half, 900–2300): the
best global offset is −10 and gains 0.01 mnats/move; per time control 1.2 mnats/move; per time control × year
2.1 mnats/move. Against a mean of about −1,380 mnats/move this is noise-level. The offsets are real but cost
almost nothing to ignore, because the likelihood curve is flat within ±100 of the nominal rating.

v0d is a good Maia-3-level policy with a better think-time head, not yet a personal model.

Diagnosis (`data/v0/runs/v0d/log.jsonl`): the training KL climbs from 2.0 to 2.96 nats and stays under the
3.0 free-bits floor for all 32.8k steps. Below the floor the KL is free, so the penalty is not what limits z.
The decoder finds little more to read from one half-game. The z-swap gain is 0.0015 nats per move, about
0.05 nats per half-game. That matches the baselines above: one game of surface style predicts rating at R²
0.18, ten games at 0.52, and re-identification needs a player's full history. Lowering the KL weight or raising
free bits would only add noise to z. The stronger fix is evidence: encode z from several games by the same
player (a set encoder over k games, k = 5–20) and make the persistence check per player across many games, as
in the style-persistence study.

**What this means for a Doppel.**

1. Judge "weird" against the player's own band's book, not a pooled one: lines are band-specific (book exit
   8.9 → 13.0 plies in own-band books, and 6.8 for a ≤900 game in a 2300+ book).
2. Treat ratings below 2000 as noisy, and 1500–1599 as close to uninformative (the provisional start).
   Rating-conditioned models (Maia-3's Elo inputs, our `elo_self`/`elo_opp`) should see an uncertainty signal.
3. Style is very stable within a player (opening choice split-half ρ ≈ 0.97, think 0.93) and transfers
   across blitz speeds. A per-player z should re-identify at least as well as surface style plus rating (81%
   top-1 among 1,000) and beat the games-needed curve (R² 0.52 for rating at 10 games).
4. The previous game's result is a cheap input for the think-time head: after a loss, players move 0.12 s
   faster and lose within 40 plies 1.4 pt more often.
5. Clock state changes strength: in mutual time trouble, rating matters about half as much (Elo scale 1,152
   vs 607).

**Data limits found.** The public dumps hold rated games only, so true casual games are not available. Every
BOT game in them is BOT vs BOT, so BOT-vs-human contrasts need another source. TWIC mixes online events
(Titled Tuesday) with over-the-board ones, so an over-the-board vs online split needs an event/site classifier.

## 9. Adding a capability

1. Write its section here first: target definition, metrics, and which priors, heuristics and machines it is
   scored against.
2. Write the label code in `lab/doppel_lab/` as a sidecar to the existing artifacts, joined on `game_id`, with
   tests in `lab/tests/`.
3. Record the prior and heuristic baselines in `lab/results/` before any neural work.
4. Add machine baselines or proxies.
5. Only then add a head to the Doppel model.

## 10. Literature

Prior work we build on, with the one-line finding that matters to us. "Read" means someone on the project has
read the source in full; "abstract" means only its abstract or a summary so far.

### Human-imitation models

- **Allie: think time, resignation and moves from one model.** Zhang, Jacob, Lai, Fried, Ippolito, *Human-aligned
  Chess with a Bit of Search*, ICLR 2025, [arXiv:2410.03893](https://arxiv.org/abs/2410.03893). A 355M decoder
  over the move sequence with policy, think-time and value heads, plus a `<resign>` token; adaptive search scales
  with predicted think time. Think-time r = 0.697 (raw seconds), resign TPR 86.4% at FPR 0.1%. Read.
- **Allie 2.1: a clock-reading successor that beats Maia-3.** [github.com/y0mingzhang/allie](https://github.com/y0mingzhang/allie)
  (MIT), weights [`yimingzhang/allie-2.1`](https://huggingface.co/yimingzhang/allie-2.1). 9.2B mixture of experts,
  1.2B active, every Lichess month from 2017 to 2026 and every time control. Clock inputs are Fourier features of
  log seconds for both clocks and the previous think time; think time is a 63-bin side head. Resigning is a fitted
  hazard, not a token. README and bot guide read.
- **Maia-3: our move-choice teacher.** *Chessformer: A Unified Architecture for Chess Modeling*, [arXiv:2605.19091](https://arxiv.org/abs/2605.19091), weights
  [MaiaChess/maia3](https://huggingface.co/collections/MaiaChess/maia3). Blitz only, no clock input. Used, not
  yet read in full.

- **Maia (original): human move prediction, tuned by rating, and blunder prediction.** McIlroy-Young, Sen,
  Kleinberg, Anderson, *Aligning Superhuman AI with Human Behavior: Chess as a Model System*, KDD 2020,
  [arXiv:2006.01855](https://arxiv.org/abs/2006.01855). Read.
  - Weakened engines do not imitate humans. Depth-limited Stockfish matches 33–41% of human moves, and every
    depth predicts strong players better than weak ones. Leela versions are flat across ratings. Maia, one
    search-free network per 100-point band, reaches 46–53% and peaks near its own band.
  - Search hurts prediction (10 rollouts cost 5–10 points); 12 plies of history add about 3 points.
  - Evaluation protocol, later copied by Allie: drop the first 10 plies, every move made with under 30 s left,
    and bullet games. Our think-time and resignation targets must keep those moves, so we report both: our
    full set, and the Maia filter when comparing move matching with published numbers.
  - Blunder prediction: a blunder is a drop of ≥ 10 points of win probability, from an empirical table of
    win rate by Stockfish cp in 10 cp steps (not Lichess's logistic curve). On a balanced test set, a residual
    CNN reaches 67.7% accuracy from the board alone and 71.7% with metadata (ratings, clock fractions, and the
    cp score). Collective version: positions seen ≥ 10 times, predicting whether > 10% of players blunder,
    76.9%.
  - Units: 10 points of win probability on a 0–1 scale is a drop of 0.2 in Lichess winning chances (−1..1),
    Lichess's "mistake" level, not its 0.3 "blunder".
  - Open direction they name: finding distinctions among players with the same rating. That is our per-game z.
- **Difficulty predicts human error better than skill or time.** Anderson, Kleinberg, Mullainathan, *Assessing
  Human Error Against a Benchmark of Perfection*, KDD 2016 ([arXiv:1606.04956](https://arxiv.org/abs/1606.04956));
  journal version ACM TKDD 11(4), 2017. Read.
  - Ground truth from 6-piece tablebases: a blunder is a move that worsens the game-theoretic result. Data: 24.6M
    instances from FICS 3-minute games, plus 880k from grandmaster games.
  - **Blunder potential** β(P) = blunders / legal moves: the chance a random move blunders. The blunder rate rises
    smoothly with β. Players act as if they are c times likelier to pick a non-blunder (c ≈ 15 on FICS, ≈ 100 for
    grandmasters).
  - On a balanced test set, difficulty features reach 0.75 accuracy; rating alone 0.55, time left alone 0.53, and
    neither adds anything once difficulty is in. A step of 0.2 in β matters more than 600 rating points. With the
    exact position fixed, rating still gives 0.62: skill matters within a position.
  - Blunder rate is flat above about 10 s left in 3-minute games and rises sharply below.
  - **Longer thinks predict more blunders**, at every time left and blunder potential and within fixed positions.
    The authors read it as "the extra time suggests they didn't know what to do". Think time and error share a
    cause, which is direct evidence for modeling them together.
  - Some positions are skill-anomalous: stronger players blunder more there.
  - Scope limits: endgames with at most 6 pieces only, FICS 3+0 in 2016, decision trees on hand features.

### Evaluation trajectories and per-game features

- **Volatility-weighted accuracy (Lichess).** [lichess.org/page/accuracy](https://lichess.org/page/accuracy).
  Win chances from centipawns; the game is cut into sliding windows whose size depends on game length, and each
  window's volatility is the standard deviation of win chances in it. The direct precedent for a volatility
  feature. Read.
- **Game volatility as one number.** jk_182, [Quantifying volatility of chess games](https://chessenginelab.substack.com/p/volatility)
  ([Lichess copy](https://lichess.org/@/jk_182/blog/quantifying-volatility-of-chess-games/H6MWvX98)) and
  [Comparing ways to calculate accuracy](https://chessenginelab.substack.com/p/different-accuracies). Weaker
  players have more volatile games, so volatility is partly a rating signal. Abstract.
- **Sharpness from win/draw/loss.** jk_182, [Evaluating sharpness using Lc0's WDL](https://chessenginelab.substack.com/p/evaluating-sharpness-using-lc0s-wdl),
  [Calculating the sharpness of different players](https://chessenginelab.substack.com/p/calculating-the-sharpness-of-different).
  A per-position "how easy is this to get wrong" score from win/draw/loss, borrowed from Lc0's WDL contempt
  code. Depends on search nodes. Maia-3's win/draw/loss could supply it without Stockfish. Abstract.
- **Path signatures over a game.** *Chess Signatures of Play*, 2026 preprint, [arXiv:2606.18544](https://arxiv.org/abs/2606.18544).
  Each game is a path over plies with eval, complexity (spread of top-k engine evals) and time channels; the
  signature turns the whole path into fixed-length features. The closest prior work to a trajectory-to-features
  analyzer. Abstract.
- **Long-range memory in move sequences.** Perotti et al., *Memory and long-range correlations in chess games*,
  Physica A, 2013 ([record](https://catalogo.biblio.unc.edu.ar/Record/rdu-unc.28101/Details)). Rescaled-range
  and detrended fluctuation analysis; stronger players show correlations over longer spans. Abstract.
- **Rating from a game, move by move.** *Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM*, [arXiv:2409.11506](https://arxiv.org/abs/2409.11506), 2024. CNN plus
  bidirectional LSTM over positions and clock times, on 1.2M Lichess games. Abstract.

### Skill and complexity from engine analysis

- **Position complexity raises error rates.** Guid & Bratko, *Computer analysis of world chess champions*, ICGA
  Journal, 2006, and later stronger-engine replications ([ChessBase summary](https://en.chessbase.com/post/using-che-engines-to-estimate-human-skill)).
  Players err more as positions get more complex. Abstract.
- **Intrinsic ratings: sensitivity and consistency.** Regan & Haworth, *Intrinsic Chess Ratings*, AAAI 2011,
  [paper](https://ojs.aaai.org/index.php/AAAI/article/view/7951); Regan, [Intrinsic Ratings Compendium](https://cse.buffalo.edu/~regan/papers/pdf/Reg12IPRs.pdf).
  Two fitted parameters per player or event describe how move choice falls off with engine value; a
  hand-crafted ancestor of a per-game style latent. Abstract.

### Clock and tilt

- **Do mistakes provoke new mistakes?** IEEE Transactions on Games ([PDF](https://www.nes.ru/files/faculty/Do-mistakes-provoke-new-mistakes-Evidence.pdf)).
  Tests within-game tilt on tournament data; the counterpart to our shock analysis (§5). Abstract.
- **Mistake rate by clock.** jk_182, [How does the clock impact the rate of mistakes?](https://chessenginelab.substack.com/p/mistakes-clock);
  a 68M-game study of the Lichess low-time alarm ([forum post](https://lichess.org/forum/redirect/post/1SL2qHTI)).
  Community analyses, not peer-reviewed. Abstract.
