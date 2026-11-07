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
0.067). Fits on the train split (`--fit train`) follow once the train evals finish. Policy-implied machine
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
grows about 0.5 ply per doubling of the corpus. Position (transposition) keys add about 2 plies.

**Upsets** (gap ≥ 300, 38,930 heldout games). The underdog wins about 21% at every gap from 300 to 700,
against an Elo expectation of 12% → 1.4%, so large Lichess gaps mostly mean a mis-rated account. From the
first 30 plies, upsets are predictable: AUC 0.60 (gap and time control) → 0.78 (+ early T0 features).

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
rating scale widened by about 600 points between the p25 and p95 tiers, which is why tiers are percentiles.

**Populations** (`doppel_lab/populations.py`, rating-matched by coarsened exact matching on year, time control,
own and opponent rating). Arena tournament vs lobby games, a competitive-vs-casual proxy, differ very little:
+1.2 plies, 0.04 s faster per move, 0.3 pt fewer weird moves. At matched rating, titled players (blitz, 6,670
sides) score 2.0 pt above their Elo expectation (z = 3.2) and play 0.5 pt fewer weird moves (z = −3.1).

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
