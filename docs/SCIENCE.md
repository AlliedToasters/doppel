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

**Time management by rating** (`doppel_lab/clock_curve.py`, 2026-10 corpus, 3+0; clock left after own move N,
as a share of the base). Weak players spend early: after move 20, a 600 has 49% left and a 1800 has 71%. From
1800 up the early curve flattens (93% after move 10, from the book). Deep into long games, strong players run
lower: after move 50, 10–11% left at 2200–2600 against 17–18% at 1400–1600, and 47–48% of 2400+ sides are
under 10% of the base. Weak players are short of time from the start; strong players save time early and spend
it in long, complex games. The pattern is U-shaped in rating, so a think head conditioned on rating alone must
not be monotone.

**Long thinks cluster** (`doppel_lab/think_ac.py`, 6.0 M own moves, 1500 train table). Maia-3 entropy and
top-move probability, clock, ply and the side's own pace in the game explain R² 0.30 of log think. The rest
is autocorrelated: residual correlation with the side's previous move is +0.153, then +0.143 at lag 2, +0.114
at 3, +0.068 at 5 and 0 by 10. Players have short "slow" and "fast" spells lasting about five moves, which no
position feature explains. A think-time head needs recurrent state, or the recent think history (the
`think_*_ewma`/`_roll` T0 channels), to reproduce them. Sampling each move's time independently would
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
pre-move predictor yet: AUC 0.932, AP 0.541. Blunder potential β (the share of legal moves that blunder) scores
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
head should see the clock difference, not only the eval. Policy-implied machine
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
These are the bars the Doppel z (`bench_doppel_<run>.json`, pending) and the Maia rating sweep must clear.

On TWIC (move-only, `lab/results/bench_maia3_twic.json`), Maia's embedding wins every task. FIDE rating R²
rises from 0.094 to 0.140 and year R² from 0.036 to 0.064. Era 2021+ AUC goes from 0.619 to 0.688, online vs
OTB among fast games from 0.630 to 0.703, fast vs classical OTB from 0.564 to 0.652, and junior from 0.563 to
0.581. Without clocks the surface has little to work with, and move quality carries the signal. Online play
and fast OTB play are visible in the moves themselves.

The v0d Doppel z (`doppel_lab/bench_doppel.py`, `lab/results/bench_doppel_v0d.json`; the posterior mean per
side and game half, 8 dims) is far weaker than both. Rating R² is 0.207 (trees; 0.144 linear) against 0.492 for
the surface and 0.532 for Maia's embedding. Top 5% AUC is 0.714, titled 0.599, arena 0.686, era 0.576. Added
to the surface it changes nothing (rating 0.493, arena 0.715). Its 8 dims hold some rating and clock-style
information, and none of it is new.

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
