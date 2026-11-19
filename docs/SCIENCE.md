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
  when a baseline needs Stockfish features on every position. Blunder rates from 2,866 positions are noisy
  (about ±0.05).
- **Eval approximator (measured 2026-11-08; `doppel_lab/evalnet.py`, `evalnet_ft.py`,
  `lab/results/evalnet{,_ft}.json`).** The question: how well does a net estimate our 50k-node win chance, and can
  it stand in for Stockfish in the science? The setup:
  - Maia-3 79M with both Elos fixed at 1500.
  - Target: side-to-move wc.
  - Scored on 5,000 held-out 1500-band games (347k positions), and out of distribution on the six-band corpus's
    12k Stockfish games (829k positions).
  - Probes on frozen features use 25k training games; the fine-tune trains the whole trunk plus a new head, one
    pass over 80k games.

  | Estimator | wc correlation | wc MAE | Blunder recall | Blunder precision | Blunder AUC |
  |---|---|---|---|---|---|
  | Maia-3's own W/D/L head, calibrated | 0.926 | 0.132 | 0.41 | 0.29 | 0.76 |
  | ridge on the frozen pooled features | 0.948 | 0.112 | 0.47 | 0.40 | 0.83 |
  | MLP on the frozen features | 0.966 | 0.077 | 0.50 | 0.52 | 0.87 |
  | **fine-tuned trunk** (80k games) | **0.979** | **0.056** | 0.59 | 0.67 | 0.93 |
  | fine-tuned, six-band corpus (out of distribution) | 0.976 | 0.058 | 0.58 | 0.63 | 0.93 |

  The fine-tuned net sits between 100 and 1,000 nodes of Stockfish (table above). It matches 1,000 nodes on
  eval accuracy but not on blunders (recall 0.59 vs 0.65, precision 0.67 vs 0.78). It runs at 11.6k
  positions/s on one GPU. That is about 20× the 50k-node labeller and about the same as 1,000 nodes on 28 cores.
  **Data scaling** (fine-tuned on 20k, 40k and 80k games): wc MAE is 0.061, 0.058 and 0.056, gaining about
  0.003 per doubling, while blunder recall stays at 0.58–0.59. More labels sharpen the eval slowly and don't touch
  blunders. A blunder is a tactic, and a static net at this size doesn't see it, the same wall the static NNUE
  hits (recall 0.06).
  A longer run (all 100k labelled games, 3 epochs, 3.75× the compute of the 80k pass; `scaling.g100k_e3`) gets
  wc MAE 0.052 (2.62 win%), blunder AUC 0.936 and recall 0.62 at the 0.3 threshold (precision 0.67). That is the
  first recall gain, but small: 1,000-node Stockfish is still ahead (0.65 / 0.78). Compute buys eval accuracy
  faster than it buys blunders.

  **Science test:** game_science's per-ply table rebuilt from the net's evals, then the same game_mean rating
  model on the 12k games:

  | Eval source for T2 | T2 MAE | T2 + rating-free T1 |
  |---|---|---|
  | none (T0: moves, clocks) | 242 | 230 |
  | frozen MLP | 232 | 223 |
  | fine-tuned net | 227 | 219 |
  | Stockfish 50k | 222 | 215 |

  Per-side ACPL from the fine-tuned net ranks players like Stockfish's (Spearman 0.85; per side-game mean loss
  0.87). The answer to the question: a net recovers about 75% of Stockfish's rating signal over T0, and with T1 it
  beats Stockfish T2 alone. That makes it a usable T2-lite for any position set too big to label (the 49M v0
  positions, a whole month of games), computed in the same GPU pass as T1. It is not a substitute for blunder
  labels or for the final numbers in a write-up: keep Stockfish 50k for labels and headline results, and use the
  net for screening at corpus scale.

  **Screening test (2026-11-09; `evalnet.py screen`, `evalnet_ft.json` "screen36k").** The fine-tuned net
  labelled all 36k corpus games (2.48M positions in 3.5 minutes; Stockfish has 12k of them). Same rating model,
  5-fold CV on all 36k, scored on all 36k and on the 12k Stockfish games (same CV predictions, filtered):

  | Features | 36k, net evals | on the Stockfish 12k | Stockfish on its 12k |
  |---|---|---|---|
  | T0 | 240 | 239 | 242 |
  | T0 + T1 | 227 | 225 | 230 |
  | T2 | 223 | 223 | 221 |
  | T2 + T1 | 215 | **214** | 214 |

  With 3× the games, net T2 closes the gap to Stockfish T2: 223 vs 221, and 214 vs 214 with T1. At this corpus
  size the rating model is data-limited as much as label-limited, so cheap labels on more games are worth as much
  as expensive labels on fewer. For rating-style science, the net on a whole corpus is a valid substitute;
  blunder-level claims still need Stockfish.
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
in `v0_data.py` carries the median and a 0.7-decay EWMA of the side's recent think fractions. Sampling each move's time independently would
flatten them.

**Checked 2026-11-08** (`doppel_lab/think_spells.py`, `lab/results/think_spells_v0d.json`). The data is 2,000 val
games: 132k own moves at ply ≥ 10, with real positions and clocks, and z drawn once per side from the prior. The
residual is log1p(think s) after entropy, clock and ply, minus the side-game mean.

| Within-game residual correlation | lag 1 | lag 2 | lag 3 | lag 5 | mean log1p(s) |
|---|---|---|---|---|---|
| human | +0.148 | +0.136 | +0.104 | +0.053 | 1.48 |
| v0d sampled, tempo from human thinks (as trained) | +0.137 | +0.104 | +0.081 | +0.033 | 1.49 |
| **v0d sampled, tempo from its own samples** (a Doppel playing) | **+0.169** | +0.121 | +0.085 | +0.035 | 1.58 |
| v0d median time, own tempo (no sampling) | +0.464 | +0.377 | +0.298 | +0.175 | 1.54 |

Fed its own sampled times, v0d reproduces human spells. Lag 1 is slightly sticky (+0.17 vs +0.15) and the tail
decays a little faster. Its times run about 10% slow (+0.10 in log1p). Two rules for shipping the head:
sample the time, never the median, because the median plus the tempo feedback locks into spells three times
too strong; and feed the bot's own sampled times back into tempo_feats. The live bots don't use the think
head yet (`backend/infer` returns no think time).

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

**Verbatim blunders (2026-11-16, `doppel_lab/verbatim_blunders.py`, `lab/results/verbatim_blunders.json`).** Rates
ask how often a player blunders; this asks whether a model names the exact blunder. Every Allie test position in
the 1100, 1500 and 1900 buckets (k = 40 context games), each played move graded by Stockfish (20k nodes, loss ≥ 0.2).
Top-1 on the played move, game bootstrap:

| Bucket | Blunders | Maia-3 | Doppel (v0g) | Doppel − Maia-3 | Learned lookup − Maia-3 |
|---|---|---|---|---|---|
| 1100 | 12,044 | 22.2% | 22.6% | +0.45 [+0.17, +0.75] | — |
| 1500 | 5,143 | 22.2% | 22.2% | +0.04 [−0.33, +0.38] | +0.60 [+0.23, +1.00] |
| 1900 | 8,901 | 22.9% | 22.9% | +0.07 [−0.19, +0.28] | — |

A population model already puts its top move on more than one blunder in five: most blunders are the natural
human move. Personal gains are larger on errors than on good moves (1100 Doppel: +0.19 on moves losing < 0.05,
+0.45–0.50 on mistakes and blunders; 1500 lookup: +0.27 vs +0.60), so what is personal about a player shows up more in
their errors. The lookup's edge fits the revisit result in §8 (habitual blunders persist): it gets 68 blunders right
that Maia-3 misses and misses 37 that Maia-3 gets.

## 8. Game-level, book, population and era baselines

**Why.** These are the surface-feature scores a learned match encoder (Maia embeddings, the Doppel z) has to
match or beat on the same splits (§2). Every result here is CPU-only.

**Game level** (`doppel_lab/game_science.py`, 36k human games, `lab/results/game_science.json`). From 1100 to
2100: eval volatility 4.49 → 3.72, ACPL 69.8 → 46.4, blunder rate 8.0% → 4.6%, first blunder at move
11.6 → 19.9. Lead changes stay flat at about 2.3 per game. Stronger players move faster (median think
3.6 → 2.4 s) but reach time trouble more (3.2% → 5.7%). Rating from one game: MAE 281 (time control only),
240 (T0), 223 (T2), with strong shrinkage to the mean.

**Finding, envelope features (null).** (`doppel_lab/envelope.py`, the same 12k games with evals,
`lab/results/envelope.json`, 2026-11-08.) The question: do shape, timing or difficulty features of the eval
trajectory add rating signal beyond the T2 aggregates? Five families were tested, each added to the T2 game_mean
model:

- quiet-move loss: win-chance loss where local volatility is under 2 win% points. This stands in for
  sharpness, which is not stored for this corpus.
- underwater area: mean and RMS drawdown, and the share of own moves more than 10 points underwater.
- quality band: burstiness of win-chance loss, spikes above the causal upper band, and band width per half of
  the game.
- MACD gap: the mean |fast EMA − slow EMA| of White's win% (α 0.5 vs 0.1), and its sign changes.
- time axis: the features above, weighted by seconds spent instead of by move. This covers loss per second
  thought, time-weighted underwater and volatility, win% swing per minute and time-weighted game volatility.
  Before this run, every transform used the move as its axis.

| Feature set | MAE | R² |
|---|---|---|
| T2 baseline | 221.8–222.1 | 0.376 |
| + any single family | 221.8–222.4 | 0.375–0.377 |
| + all five (98 features) | 222.1 | 0.377 |
| envelope and time features alone (32) | 245.2 | 0.262 |

The baseline moves about 0.3 Elo between identical runs, so none of the families changes the score. Within
a band, the new features correlate 0.3–0.76 with ACPL, the MACD gap correlates about 0.5 with volatility, and
no feature correlates more than |0.03| with rating. One game's eval trajectory appears to be used up as a
rating signal at this model size. Further gains would have to come from outside it: Maia-3's view of the
position (real sharpness is still untested) or more games per player. A side result: first-half band width
predicts second-half band width at +0.055 within a band, close to the +0.063 persistence of experiment zero.
It carries the same confound, because it includes skill differences between players inside a band.

**Finding, sharpness: rating-free Maia beats the T2 ceiling; sharpness-gated Stockfish features don't.**
(`doppel_lab/sharpness.py rating`, `lab/results/sharpness_rating.json`, 2026-11-08.) This follows up the envelope
study's open thread on real sharpness. Maia-3 79M was run over all 36k games of the six-band human corpus with
**both Elos fixed at 1500** (`teacher.py --fixed-elo`), so its outputs carry no rating input. Sharpness is
Maia's policy entropy in the position. Each side gets two new feature sets:

- T1: the played move's mean log-prob, top-1 agreement and mean entropy faced; log-prob in sharp positions
  (top entropy third) and in quiet ones (bottom third); the miss rate in obvious positions (Maia top move ≥ 0.8);
  and steering (the entropy handed to the opponent beyond what the faced position predicts).
- Sharpness-gated Stockfish: win-chance loss and blunder rate in sharp and in quiet positions, loss in obvious
  positions, and time allocation (Spearman of think fraction against entropy).

| Feature set (game_mean model, 5-fold CV) | Games | MAE | R² |
|---|---|---|---|
| T0 (moves, clocks) | 12k with evals | 242.4 | 0.272 |
| T0 + T1 (Maia at 1500) | 12k | 230.1 | 0.332 |
| T2 (Stockfish) | 12k | 221.4 | 0.378 |
| T2 + sharpness-gated Stockfish | 12k | 222.1 | 0.376 |
| **T2 + T1** | 12k | **214.2** | **0.413** |
| T2 + T1 + gated | 12k | 215.3 | 0.408 |
| T0 | all 36k | 240.5 | 0.281 |
| T0 + T1 | all 36k | 226.9 | 0.347 |

Gating Stockfish loss by sharpness adds nothing, which matches the envelope null. Maia's view of the moves adds
7 Elo of MAE beyond everything Stockfish gives. That is 20× the run-to-run noise and the first gain over T2 since
game_science. Without Stockfish, T1 recovers about 60% of the T0 → T2 gain. The strongest single T1 features
(Spearman with rating): mean entropy faced +0.14 (stronger players play sharper positions), steering +0.12 and
log-prob in sharp positions +0.11, against −0.14 for Stockfish loss in sharp positions. Agreement with a
fixed-1500 Maia is only weakly monotone in rating (log-prob +0.09), so the gain comes from *where* the
agreement happens, not its level.

**Finding, sharpness as style: steering is a stable trait that doesn't pay.** (`doppel_lab/sharpness.py style`,
`lab/results/sharpness_style.json` and `_min8.json`.) This uses the 1500-band train table (the 1460–1640 Elo
range, 100k games with Stockfish and Maia-3 per move): 873 players with at least 16 games, split by odd and even
games. Spearman correlations, with mean rating regressed out (rating barely moves any of them):

| Trait | Split-half ρ | Half A → half B performance | → opponents' blunder rate |
|---|---|---|---|
| steering (entropy handed beyond expected) | +0.42 | +0.02 | −0.04 |
| time allocation (think vs entropy) | +0.40 | −0.02 | 0.00 |
| Maia log-prob of own moves | +0.43 | −0.01 | −0.02 |
| mean entropy faced | +0.26 | +0.02 | −0.04 |
| own blunder rate | +0.16 | +0.03 | +0.14 |
| opponents' blunder rate | +0.16 | +0.03 | +0.16 |
| volatility steering, sharp-minus-quiet loss, obvious misses | 0.00–0.08 | ≈ 0 | ≈ 0 |

Performance (score minus Elo expectation) itself has split-half ρ +0.09 at 16 games, so a trait can't predict
it much better than this. With 5,811 players at ≥ 8 games, the halves shrink and steering reads +0.26 and
allocation +0.30, close to the Spearman–Brown expectation. Steering correlates only −0.28 with own Maia log-prob,
so it is not just "plays unusual moves". A player who sharpens the game does not make opponents blunder more and
does not outscore their rating. Opponents' blunder rate is persistent (+0.16), but it travels with the player's
own blunder rate (ρ 0.55 within player): their games are messy on both sides. For a Doppel, steering and time
allocation are style dimensions z should carry: as stable as opening choice within a band, and invisible to
rating.

*At scale (2026-11-09; `sharpness style --table moves_f1_T2net_1500_train`,
`lab/results/sharpness_style_moves_f1_T2net_1500_train.json`).* The eval net labeled all 700k v0 train games
(§3.1), so the same study runs on 26,757 players with at least 16 games, 31× the players. The eval-based traits
(volatility steering, sharp-minus-quiet loss, blunder rates) now use net evals; steering, allocation and Maia
log-prob don't use evals at all.

| Trait | Split-half ρ, 873 players | 26,757 players | Half A → half B performance |
|---|---|---|---|
| time allocation | +0.40 | **+0.51** | +0.007 |
| steering | +0.42 | **+0.49** | +0.003 |
| Maia log-prob of own moves | +0.43 | +0.46 | +0.001 |
| mean entropy faced | +0.26 | +0.34 | +0.004 |
| own blunder rate | +0.16 | +0.25 | 0.000 |
| opponents' blunder rate | +0.16 | +0.20 | −0.014 |

The finding holds and tightens. Steering and allocation are the most stable traits measured, and with 27k
players the predictive null is now precise: every cross-half performance correlation is within ±0.015.
Performance itself has split-half ρ −0.009, so at 1500 nothing about a player predicts beating their rating:
Lichess ratings already absorb it. Steering slightly *lowers* opponents' later blunder rate (−0.07): sharper
positions don't make opponents crack.

**Finding, per-player repeated positions: a player's own history beats Maia by 0.93 nats where it exists.**
(`doppel_lab/player_markov.py`, `lab/results/player_markov.json`; the user's per-player sparse Markov-chain
idea.) Each player's empirical next-move distribution at positions they reach more than once, on the 700k v0
1500-band train games. Position = Polyglot key of the decision position (transpositions merge), plies 0–79, one
visit per game. For every visit with an earlier visit by the same player, the played move is forecast online
(earlier games only) by Maia-3 alone, and by the player's earlier moves there shrunk toward Maia:
p = (n_prev(move) + α·p_Maia(move)) / (N_prev + α), with α = 0.5 best on the grid.

- **Coverage:** 2.39M of 42.9M visits (5.6%) are repeats. 73,786 of 126,458 players have one or more repeats,
  21,743 have 30 or more. Repeats are openings: median ply 2, 90% at ply ≤ 6, and only 54k at ply ≥ 10.
- **Gain:** Maia's log-likelihood at these visits is −1.47 nats; the shrunk personal history gets −0.55, a gain
  of **0.93 nats per visit**. Players replay a move they played there before 90% of the time; Maia gives that
  move 0.39 on average.
- **It grows with history:** 0.66 nats after one earlier visit, 1.04 after 3–4, 1.16 after 9–16.
- **It persists past the opening:** 0.95 at plies 1–9, 0.36 at plies 10–19, 0.21 at 20–39 (thin there: 651
  visits). It is largest at the most common positions (1.07 nats at positions with 100k+ visits), where
  personal repertoire is the whole story.
- **It is a stable per-player trait:** among 11,675 players with 60+ scored visits, 99.4% gain, median 0.87
  nats; split-half ρ of the per-player gain is **+0.89** (9,667 players, odd vs even games).

For scale: spread over all positions, 0.93 × 5.6% ≈ 0.052 nats per position, against v0f's whole
personalization gain of 0.0005 nats (ce_post − ce_teacher) and its z-swap gain of 0.0012. A set-encoder z can't
memorize "this player plays 3.Bb5 here", and this is where most of a player's predictable individuality lives at
1500. Use (revised 2026-11-11, below): the player's own history at repeated positions is a sparse,
player-level ground truth for evaluating a model, not a component to play from. Both checks are below
(repertoire vs z).

**Finding, repertoire vs z: z holds about 1% of what a player's own history shows.** (`doppel_lab/repeat_v0f.py`,
`lab/results/repeat_v0f.json`, 2026-11-10.) Same visits and protocol as player_markov, on the valrich split
instead of train: 600 held-out players' first 48 band games, which no v0f run saw. That gives 66,675 repeat visits
by 3,172 players (the 600 plus opponents met twice), 98% of them before ply 10. v0f's z is the posterior mean from
the player's 40 most recent earlier games (all of them if fewer). Gains over Maia-3 at the real Elos (−1.451
nats per visit), with player-bootstrap 95% CIs:

| Model | Gain, nats per visit | Share of the history gain |
|---|---|---|
| v0f, prior z (no player) | +0.0027 [0.0015, 0.0038] | 0.3% |
| v0f_quarter, posterior z | +0.0043 [0.0031, 0.0054] | 0.5% |
| v0f_half, posterior z | +0.0066 [0.0053, 0.0081] | 0.7% |
| v0f, posterior z | +0.0108 [0.0088, 0.0128] | 1.2% |
| own history shrunk toward Maia (α 0.5) | +0.905 [0.872, 0.937] | 100% |
| own history shrunk toward v0f | +0.906 [0.873, 0.946] | 100% |

The history gain holds on held-out players (+0.905 vs +0.93 on train), so it's not an artefact of the train
split. z captures 1.2% of it. Shrinking toward v0f instead of Maia adds +0.0009, so z and the history overlap
almost completely, and the history is all of it. z's gain grows with every context-size and N_prev bin
(+0.0075 at N_prev = 1, +0.017 at 9–16) but stays two orders of magnitude short. Across the three data-scaling
runs it rises ×1.5 per doubling of games (about N^0.65). At that rate, closing the gap would need roughly 9
more doublings, about 500M games. z does not hold the repertoire. Most of the gain is in the opening (ply 0–9: +0.91; 10–19: +0.36),
which is why the history is used as an evaluation ceiling below rather than as a per-player book to play from.

**Finding, v0f at 1100 and 1900: z helps less as rating rises; the repertoire gap holds at every band.**
(`v0e_train --band 1100/1900`, runs v0f_1100 and v0f_1900; `lab/results/v0_v0f_{1100,1900}.json`,
`repeat_v0f.json`; 2026-11-11.) Same recipe as v0f (`--w-con 0.1`, 32,808 steps, init from v0d) on 700k train
games per band, with each band's own Maia-3 teacher, context vectors and 600 held-out valrich players:

| Metric (own band's valrich) | 1100 | 1500 | 1900 |
|---|---|---|---|
| ce_teacher (nats) | 1.424 | 1.334 | 1.239 |
| ce_post − ce_teacher | −0.0023 | −0.0013 | −0.0006 |
| z-swap gain (nats) | 0.00118 [0.00097, 0.00139] | 0.00115 [0.00090, 0.00143] | 0.00062 [0.00040, 0.00084] |
| KL per game / active units | 0.88 / 23 | 0.86 / 28 | 0.96 / 26 |
| top-1 teacher → post | 0.537 → 0.538 | 0.556 → 0.556 | 0.583 → 0.584 |
| think r, prior → post | 0.669 → 0.683 | 0.697 → 0.708 | 0.702 → 0.712 |
| resign AUC (post) | 0.972 | 0.977 | 0.983 |
| Repeat visits: history gain over Maia | +0.794 | +0.905 | +0.927 |
| Repeat visits: v0f posterior gain | +0.0089 [0.0069, 0.0109] | +0.0108 [0.0088, 0.0128] | +0.0041 [0.0028, 0.0054] |
| z's share of the history gain | 1.1% | 1.2% | 0.4% |

Three patterns. First, the gain over the teacher halves with each 400 points (−0.0023, −0.0013, −0.0006), and
at 1900 the z-swap gain is half of 1100's and 1500's, though z carries slightly more KL. Maia-3 already fits
1900 players better (ce 1.24 vs 1.42), and what's left per player is harder for z to use. Second, the
repertoire signal grows with rating (+0.79 → +0.91 → +0.93 nats per repeat visit): stronger players are more
consistent at positions they've been to before, while z captures a smaller share. Third, the conclusion of
"repertoire vs z" holds at every band: z gets about 1% of the history gain or less, and shrinking toward v0f
instead of Maia adds ≤ 0.002. A per-player opening memory matters most for the strongest players. The think
and resign heads work at every band with no tuning (think r gain +0.010 to +0.014).

**Finding, repertoire memory: the best history estimate is a plain count table; use it as an evaluation
ceiling, not at inference.** (`doppel_lab/repertoire_memory.py`, `lab/results/repertoire_memory.json`; per-visit
log-probs from `repeat_v0f --dump`.) The history estimate is the player's earlier moves at the exact position
(R0 poskey, plies 0–79), mixed into v0f's posterior as p = (n_prev(move) + a · p_v0f(move)) / (N_prev + a). It is
scored online on the valrich repeat visits, with a and the variant fitted on half the players and reported on the
other half. Gain over v0f, nats per repeat visit:

| Variant | 1100 | 1500 | 1900 |
|---|---|---|---|
| count, a = 0.5 | +0.800 | +0.887 | +0.905 |
| recency-weighted (best λ) | +0.801 (λ 0.9) | +0.887 (λ 1) | +0.905 (λ 1) |
| last visit only | +0.742 | +0.836 | +0.858 |
| a by player consistency × N_prev | +0.806 | +0.893 | +0.916 |

The test-half 95% player-bootstrap CI at 1500 is ±0.05, so the variants don't separate except last-only, which is
worse. Players don't drift over 48 games, so every earlier visit counts equally, and a = 0.5 fits every band. A
player-consistency schedule for a adds less than 0.01. For the 600 target players (48 games each), repeat visits
are 6% of their own moves, about 2.1 per game, and they sit mostly in the first 20 plies.

Why not play from it. The coverage is sparse and opening-heavy. A Doppel reading a per-player book would copy the
player's opening and then fall back to the model, which looks faithful while modelling nothing. It also does
nothing for a player with few games, and it would hide whether z learned the player at all. Per-move averages
(0.05 nats per move vs z's 0.0013) are dominated by those few opening moves and are not a fidelity comparison.

How to use it. Repeat-visit fidelity: a model's gain over Maia at held-out repeat visits, as a share of this
ceiling (count, a = 0.5), reported by ply (0–9, 10–19, 20+) and N_prev. It complements KL-to-Maia. KL says how far a
model moved from Maia; repeat-visit fidelity says whether it moved toward what this player actually does. v0f
scores about 1% (§8 repertoire vs z). Caveats: the ceiling is noisy at N_prev = 1–2. These are the positions where
players are most consistent, so the metric says nothing about middlegame fidelity (style traits cover that).
First readings (`repeat_v0f.fidelity`, now part of `v0e_train eval`; `lab/results/repeat_v0f.json` "fidelity"):

| Run | share of ceiling | policy KL to Maia (nats) | gain per nat of KL |
|---|---|---|---|
| v0f_quarter (1500) | 0.47% | 0.0007 | 6.1 |
| v0f_half (1500) | 0.73% | 0.0011 | 6.1 |
| v0f (1500) | 1.19% | 0.0020 | 5.3 |
| v0f_1100 | 1.12% | 0.0026 | 3.4 |
| v0f_1900 | 0.44% | 0.0008 | 4.9 |

Two readings. First, the opening bias is extreme: 98% of repeat visits (65k of 67k at 1500) are at plies 0–9,
because with 48 games a player rarely reaches the same position after move 5. So as measured here this is a
first-five-moves metric. Later plies need more games per player (backlog item 5) or backoff. Second, v0f moves in
the right direction but barely moves: each nat of KL it spends at these positions buys 5–6 nats of the player's
log-likelihood, and that efficiency stays flat across the data-scaling runs while the KL grows with data. The
shortfall against the ceiling is magnitude, not direction. KL-to-Maia alone could not show this.

Backing off to near positions is open. The plan is to pick candidates by pawn structure and setup, then keep a
neighbour only where Maia's move distribution agrees, since one square for a rook or queen can change everything.
A probe of how Maia's board embedding and policy respond to one-piece perturbations comes first (next finding).

**Finding, one-piece perturbations: half of them change Maia's decision, and Maia's own features can tell which.**
(`doppel_lab/probe_perturb.py`, `lab/results/probe_perturb.json` "main", pairs in `data/probe/perturb_pairs.parquet`.)
Base positions: 4,000 v0 val positions (1500 band, plies 16–60). Each gets up to 8 perturbations, 32,000 pairs in
all: one non-king piece of either side steps to an adjacent empty square, and the board stays valid. Every pair
differs in exactly two squares, so surface distance is identical across all of them. Maia-3 79M at 1500/1500 runs
with no move history on both boards. "Changed" means the policy Jensen-Shannon divergence is above 0.1 or the top
move changes.

- 53% of one-square perturbations change the decision (mean JS 0.19). Pawns are not safe: 43% for an opponent
  pawn, 51% for one of the mover's own. The highest are the mover's knight (79%) and queen (75%). One square
  matters about half the time, whatever the piece.
- The heuristic does poorly. The number of squares the moved piece attacks differently separates changed from
  unchanged at AUC 0.63 (ρ with JS 0.31). Piece type and owner alone give a range of 42–79%.
- Maia's internal features do well. Cosine distance of the middle-layer features (layer 4 of 8, mean over squares)
  gives AUC 0.84 (ρ 0.74), and it holds within every piece type (0.82–0.86). The final-layer pooled features give
  AUC 0.88 (ρ 0.81), but they feed the policy directly, so that is partly circular. Max-over-squares at the middle
  layer is worse (0.77).

So the user's prior holds: a small difference on the board changes the decision half the time, and a surface or
attack heuristic can't say when. Maia's middle-layer representation can, well enough to rank candidates. The
backoff design follows from this: retrieve near positions from the player's own earlier games by middle-layer
distance, then keep only those where Maia's move distributions agree (JS below a threshold, and the remembered
move legal). The test after that is whether the kept neighbours' moves predict the player beyond Maia at plies 10+,
where exact repeats are 2% of visits. Caveat: random one-step perturbations are not the near positions that real
games produce, so the 53% is an upper-end figure.

**Finding, policy-filtered backoff: near positions carry the player only through the opening; past ply 20 a
neighbour's move is Maia's, not the player's.** (`doppel_lab/backoff.py`, `lab/results/backoff.json` "1500",
neighbours in `data/v0e/backoff_1500.npz`; one 90 s GPU job.) Queries: 866k own decision positions (plies 0–79) of
the 600 valrich target players, in every game after their first. Retrieval works in three steps:

1. Take the 16 nearest positions from the player's strictly earlier games, by cosine distance of Maia-3's layer-4
   features (with real move history and real Elos).
2. Keep a neighbour only if its policy Jensen-Shannon divergence to the query is at most τ and its played move is
   legal at the query. Exact repeats are excluded (backoff only).
3. Mix the kept neighbours' moves with Maia as in the count table.

The control runs the same retrieval over a partner target player's earlier games, from a fixed derangement with
matched pool size. τ, a and k are fitted on half the players and reported on the other half, refitted per ply range.

| Plies | own gain (nats/query) | own coverage | control gain | control coverage |
|---|---|---|---|---|
| 0–9 | +0.189 | 72% | −0.107 | 43% |
| 10–19 (refit) | +0.028 | 35% | 0.000 | 0.4% |
| 20+ (refit) | +0.0002 | 4.7% | 0.000 | 0.8% |
| 40+ (refit) | 0.000 | 0.7% | 0.000 | 0.1% |

The direct test is the neighbour's move matching the query's played move, at equal policy similarity:

| JS bin | 0–9 own / control | 10–19 own / control | 20+ own / control |
|---|---|---|---|
| ≤ 0.01 | 0.84 / 0.49 | 0.89 / 0.95 | 0.96 / 0.97 |
| 0.01–0.05 | 0.70 / 0.32 | 0.68 / 0.72 | 0.87 / 0.89 |
| 0.05–0.1 | 0.51 / 0.29 | 0.56 / 0.56 | 0.76 / 0.77 |
| 0.1–0.2 | 0.38 / 0.26 | 0.44 / 0.42 | 0.59 / 0.63 |

In the opening, the player's own near positions predict them far better than another player's (0.84 vs 0.49).
That is the repertoire again, reaching non-identical boards. From ply 10 on, the gap is gone: at a given policy
similarity, another player's neighbour predicts the move as well as the player's own. The 10–19 gain comes from
coverage, because the player's own lines produce more near positions, not from better matches. Past ply 20 the
fitted shrinkage goes to the maximum (a = 16), which means "use Maia". So even with a representation that can tell
which one-square differences matter, a player's move-level memory doesn't transfer to the middlegame. What a
neighbour's move predicts there is what the position (and Maia) already says.

Consequences:

1. Repeat-visit fidelity is an opening metric by nature, not only because exact repeats are rare. Its ply 10–19
   extension through backoff adds coverage but no new player signal.
2. Player-level middlegame fidelity has to be measured through style (the §8 traits: steering, allocation,
   think-time) and through how a model behaves across many positions, not through move recall.
3. That is z's job. A memory can't take it over. Caveat: 48 games per player; a corpus with hundreds of games per
   player (backlog item 5) could still find middlegame habits that transfer. Random one-step perturbations of
   positions are not involved here: all neighbours are real positions from real games.

**Finding, middlegame style fidelity: v0f's z gets the direction of a player's move-choice style roughly right,
at about 3% of its size; time allocation is the one trait z really carries.** (`doppel_lab/style_fidelity.py`,
`lab/results/style_fidelity.json` "v0f"; passes cached in `data/v0e/style_v0f_{A,B}.parquet`; one 13-minute GPU
job; `v0e_train eval --style` runs it.)

Setup:

- Players: the 600 valrich target players at 1500, 48 games each. Half A is games 1–24, half B games 25–48.
- Positions: up to 12 own positions per half-B game, plies ≥ 10, 170k in all.
- Model: v0f runs with z = the posterior from the player's earlier games ("post") and with z = the prior mean
  ("prior").
- Children: every legal move's resulting position goes through the trunk with the opponent to move. That gives
  steering (entropy handed to the opponent) and conformity (trunk log-prob of the move) as exact expectations under
  the model's move distribution. 1.7M positions of pass A give the actual traits.

The first reading was confounded by position. Scored at the player's own positions, the trait mean mostly
reflects which positions the player reaches. At plies ≥ 10, the trunk alone predicts each player's actual half-B
steering at ρ 0.72, above their own half A (0.58), and v0f post adds 0.001. So every comparison below is at the
level of move choice: the played move's value minus Maia's expectation at that position (actual), against the
model's expectation minus Maia's (post, prior).

| Plies ≥ 10, across 600 players | reliability (half B) | post ρ | prior ρ | z adds | model SD / actual SD |
|---|---|---|---|---|---|
| steering, move-choice | 0.52 | +0.17 | +0.13 | +0.03 | 0.0009 / 0.034 (3%) |
| conformity, move-choice | 0.55 | −0.11 | −0.01 | −0.10 | 0.0010 / 0.100 (1%) |
| time allocation (think head) | 0.59 (A vs B) | +0.58 | +0.52 | +0.06 | — |

Plies ≥ 20 read the same (steering +0.15 vs +0.13; conformity −0.13 vs −0.07; allocation +0.39 vs +0.34, history
0.37).

1. Move-choice steering is a real trait (split-half 0.35 within half B), and v0f ranks players on it a little
   (ρ +0.17, the ceiling is 0.72). But it spreads players 38× less than they really differ. This is the same
   pattern as the repeat-visit metric: right direction, far too little movement.
2. Move-choice conformity (how far a player's moves fall below Maia's expectation) is just as reliable, and v0f
   gets it wrong. z moves the ranking from −0.01 to −0.11 (about 2.6 standard errors from zero for 600 players; not yet bootstrapped). The model's deviations from
   Maia are largest, or aimed at less likely moves, for the players who are actually more conventional. Its size
   is 1% of the real spread, so this flags a direction error to check, not a measured harm.
3. Time allocation is where z works. Post reaches the history ceiling (0.58 vs 0.59), and z adds +0.06 over the
   prior. Part of the prior's 0.52 comes from the think head's in-game tempo features (the player's earlier think
   fractions in the same game), so z's share is the +0.06.

So in the middlegame, z carries the player's clock habits and almost none of their move-choice style. The
move-level metrics to watch for v0g are the model-to-actual SD ratio (now 1–3%) and the sign of the conformity
ρ. Raising the residual's reach (less KL pressure, larger z effect) should move the first, and the second needs a
look at what z's residual does for unconventional players.

*Follow-up: the conformity sign across five models* (2026-11-12; one 40-min job for the other four runs,
`style_fidelity --runs`; bootstrap and mechanism in `conformity_check.py`, 2,000 player resamples, plies ≥ 10;
`lab/results/style_fidelity.json` "conformity_check"):

| Run | Conformity post / prior | z adds [95% CI] | Steering z adds | Allocation post / history |
|---|---|---|---|---|
| v0f_quarter (1500, 175k) | +0.02 / +0.03 | −0.016 [−0.042, +0.008] | +0.065 | 0.54 / 0.59 |
| v0f_half (1500, 350k) | −0.06 / +0.00 | −0.066 [−0.112, −0.023] | +0.048 | 0.56 / 0.59 |
| v0f (1500, 700k) | −0.11 / −0.01 | −0.096 [−0.154, −0.040] | +0.034 | 0.58 / 0.59 |
| v0f_1100 | −0.01 / −0.01 | −0.001 [−0.073, +0.076] | +0.115 | 0.53 / 0.59 |
| v0f_1900 | +0.03 / +0.04 | −0.009 [−0.070, +0.050] | +0.017 | 0.55 / 0.58 |

The 1500 effect is real (bootstrap p < 0.001 for v0f and v0f_half) and grows with training data, but the 1100 and
1900 models, scored on their own 600 players, show none. The three 1500 runs share their target players, so this
is one player sample, not three replications. Steering and allocation gains replicate in every run.

Mechanism: it is a direction, not a size artifact. The model's E[log p_maia] falls whenever z moves mass off
Maia's likely moves, but z's push is negative for only 50–60% of players, and its size does not track actual
conformity (ρ −0.04 to +0.06). What does stand out: z moves steering and conformity together along one axis
(per-player ρ of the two z pushes −0.82 at 1500, −0.96 at 1100, +0.28 at 1900), while real players' steering and
conformity are nearly independent (−0.10, −0.22, −0.00). At 1500, z's steering push tracks actual conformity
(+0.17) better than actual steering (+0.12): z has learned who is conventional, and the decoder spends it as "steer
sharper", which costs Maia-likeness. With a z effect only 1–3% of the real spread, the policy expresses roughly
one style dimension. For v0g, the check is that the two z pushes decouple (|ρ| well below 0.8).

**Finding, v0 data scaling: z's gain grows almost linearly with games and hasn't flattened.** (`v0e_train
--train-frac 0.5` and `0.25`, runs v0f_half and v0f_quarter; `lab/results/v0_v0f_{half,quarter}.json`,
`players_pool_v0f_{half,quarter}.json`.) The v0f recipe on a random half and a random quarter of the 700k train
games (dropped games leave both the targets and the players' histories), at the same 32,808 steps, against v0f on
the full set:

| Metric (full val) | v0f_quarter, 175k | v0f_half, 350k | v0f, 700k games |
|---|---|---|---|
| ce_post − ce_teacher (nats) | −0.00062 | −0.00090 | −0.00128 |
| z-swap gain (nats) | 0.00032 [0.00025, 0.00038] | 0.00069 [0.00054, 0.00085] | 0.00115 [0.00090, 0.00143] |
| KL per half-game / active units | 0.47 / 16 | 0.69 / 23 | 0.86 / 28 |
| re-ID top-1 at k = 5 / 10 / 20 / 40 | 0.19 / 0.34 / 0.87 / 0.97 | 0.43 / 0.56 / 0.92 / 0.97 | 0.46 / 0.67 / 0.93 / 0.97 |
| rating R² at k = 40 | 0.77 | 0.74 | 0.74 |
| top-1, think r | 0.557, 0.704 | 0.556, 0.706 | 0.556, 0.708 |

The z-swap gain goes 0.00032 → 0.00069 → 0.00115 (CIs disjoint at every step). Each doubling adds more than the
last (+0.00037, then +0.00046), so the curve is not log-linear and shows no sign of flattening: over 4× the data
it rises 3.6×, a power law near N^0.9. KL and active units climb with it (16 → 23 → 28). Re-ID at 5–10 games
carries the change (k = 10: 0.34 → 0.56 → 0.67); at 20+ games all three runs re-identify, and rating R² at 40
games is flat (0.74–0.77, inside fold noise), so rating is learned from a quarter of the data. If the power law
holds, doubling the corpus gives about +0.001 nats of z-swap, twice the +0.0005 a log-linear guess from two points
gave. That is still well below the repertoire signal above (0.05 nats per position). A bigger corpus pays twice:
more games per player for z, and more repeat visits for a repertoire memory, which is the larger prize. Backlog
item 5 (a full month) is the test.

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

*Per-game uniqueness curves* (`uniqueness_curve.py`, `uniqueness_endgame.py`; `lab/results/uniqueness_curve.json`;
same probe and book, the endgame half recounted with `poskey count --cap 255`, 70 s). Each game's count of train
games sharing its exact position falls geometrically: the median count drops 2.7–2.9× per ply after ply 2
(30,587 at ply 4, 396 at ply 8, 8 at ply 12, 0 at ply 16). That rate predicts 0.66 plies of depth per doubling
of the corpus, close to the measured 0.71: the per-ply decay and the corpus multiple per ply (2.66×) are the same
number. The first unique ply has mean 15.0, SD 4.8 (q10–q90 9–21), and the spread widens with the corpus (q90
moves 0.88 plies per doubling, q10 0.50). The fall is almost monotone: 12% of in-book steps rise as a
transposition merges routes, and 73% of games have at least one rise. Pushing the median exit from 15 to 20
plies would take 2.66⁵ ≈ 130× the corpus (about 3 B games).

After going unique, a game stays unique through the middlegame and comes back only in bare endgames. Share of
post-exit positions that some train game also reached, by men left (kings included), 22.7 M book:

| Men | 25–32 | 17–24 | 11–16 | 7–10 | 4–6 | 2–3 |
|---|---|---|---|---|---|---|
| Positions known | 2.5% | 0.08% | 0.03% | 0.8% | 19% | 94% (median count 68) |
| Games reaching it that rejoin | 19% | 0.8% | 0.2% | 3.5% | 41% | 89% |

The 25–32 rejoins are move-order merges 1–2 plies after exit; that share barely grows with the corpus (17.7% at
89k games, 19.3% at 22.7 M). The endgame rejoins grow fast and saturate: 4–6 men known 0.8% → 2.2% → 5.3% →
11% → 19% for each 4× of corpus, because the space of bare endgames is small. Only 9.6% of games reach ≤ 6 men,
and 5.7% of games end in a known position (most end by resignation or flag at a median 16 men). So the curve is
an L with a late hook: exact ground truth exists in the opening and in pawn-and-king endings, and almost nowhere
between. Population counts there are not personal repeat visits, and ≤ 7 men is tablebase territory anyway.

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

*Exact-repeat memory on the benchmark's positions* (`doppel_lab/ceiling.py`, `lab/results/ceiling.json`,
2026-11-12; 12.7 M held-out 2026-10 games, 881k players, R0 key over the first 40 plies). Each visit is predicted
by the player's own modal move from their strictly earlier visits to the same position in the month, when there
are any, else by the crowd's leave-one-out modal move (non-causal, which favours the crowd). Top-1 gain of that
switch over the crowd, over all positions, with a 500-sample player bootstrap:

| Positions | Share with own history | Own top-1 | Crowd top-1, same visits | Gain, points of top-1 [95% CI] |
|---|---|---|---|---|
| Full games (849 M plies) | 6.3% | 82.7% | 50.5% | **+2.02** [2.016, 2.034] |
| Allie protocol (ply ≥ 10, 30 s cut; 630 M) | 0.68% | 74.6% | 60.1% | **+0.099** [0.097, 0.100] |

By ply: 43% of visits at plies 1–9 have own history (own 83.4% vs crowd 49.6%, +14.5 points in that range);
3.2% at plies 10–19 (+0.50); 0.15% at 20–29 (+0.005); nothing after. **The benchmark's ply ≥ 10 filter removes
about 95% of the gain that exact-repeat memory gives over a game.** On the benchmark itself the repeat-memory share
of per-player headroom is small but measurable: about +0.1 points against Maia-3's 57.1, so a personalized gain
there must mostly come from generalisation (style on new positions). Lower bound in two ways: one month of history
per player (Allie-track players can have more), and a crowd baseline that has seen the whole month.

By rating (200-Elo bucket of the two players' mean; same run, 2026-11-12). The Allie test set is stratified flat,
about 11% of positions per bucket from 800 to 2200 and 6.7% at 2400, while 2026-10 piles up at 1400–1800:

| Bucket | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | 2200 | 2400 | 2600 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Allie gain (pts) | 0.007 | 0.015 | 0.033 | 0.046 | 0.062 | 0.088 | 0.133 | 0.208 | 0.329 | 0.429 | 0.367 |
| Allie own-history share | 0.1% | 0.2% | 0.3% | 0.4% | 0.5% | 0.6% | 0.9% | 1.3% | 1.8% | 2.2% | 1.9% |
| Full-game gain (pts) | 1.05 | 1.35 | 1.64 | 1.83 | 1.96 | 2.10 | 2.24 | 2.36 | 2.42 | 2.31 | 1.82 |

**Memory gain on the benchmark grows about 60× from 600 to 2400:** strong players repeat deeper lines (own history
past ply 10 is 20× as common) and repeat them more faithfully (own top-1 87% vs 57%). Reweighted to the test set's
rating mix, the Allie-protocol gain is **+0.14 points** (vs +0.10 at 2026-10's mix); the full-game gain is flat
(2.0 either way). The expert end of the test set is where repeat memory pays, and where Maia's population policy
is least personal.

*The same on the Allie test set itself* (`doppel_lab/allie_memory.py`, `lab/results/allie_memory.json`, 2026-11-12;
first run, partial history). Each of the 884,049 protocol positions is matched by move prefix (transpositions not
merged; plies ≥ 40 count as no history) against the test player's own games started strictly before the test game
(data/allie/history, 2021-11 to 2022-05 so far, test games dropped by id). The crowd is the modal move in the 2022
base sample of non-test players (12M+ games, all of 2022-01..05, non-causal), and it covers 88% of the positions that
have own history. An uncovered position counts as a crowd miss, so that part favours the switch.

| | Own-history share | Own top-1 | Crowd top-1, same | Gain (pts over all 884k) |
|---|---|---|---|---|
| All | 1.08% | 76.8% | 52.8% | **+0.26** |
| plies 10–19 | 5.2% | 76.7% | 53.4% | +1.21 in range |
| bucket 1000 / 1600 / 2200 / 2400 | 0.2% / 0.8% / 2.4% / 2.5% | 64 / 69 / 82 / 86% | 53 / 48 / 56 / 60% | +0.02 / +0.17 / +0.63 / +0.65 |
| test game month 2022-01 / 05 / 12 | 1.1% / 1.5% / 0.8% | | | +0.31 / +0.33 / +0.17 |

Rerun 2026-11-13 with history through 2022-11 (13 of 14 months) and base 2022-01..11: own-history share 1.35%, own
top-1 76.8% vs crowd 54.2% (coverage 91%), **gain +0.31 points**. By bucket +0.05 (1000), +0.21 (1600), +0.75 (2200),
+0.74 (2400). Every test month from 2022-06 on now sits at +0.27 to +0.35, the same level as January with only 2
months of history (+0.29). A few months of history capture most of the repeat memory; more history adds little.

**Against Maia-3 itself** (2026-11-13, GPU job 29767bf207d9, `top1 --cut-clock 30 --save-positions`). Maia-3-79M on
the exact protocol: **57.09% top-1 on 884,049 positions**, matching the published 57.1. On the 11,945 positions
(1.35%) where the player has an earlier visit, Maia-3 gets 61.5% and the player's own modal earlier move gets 76.8%.
Both are right on 6,547, only memory on 2,630, only Maia on 797. Switching to own memory whenever it exists:
**57.30% (+0.21 points)**. Requiring ≥ 2 / 3 / 5 earlier visits makes memory more accurate (82 / 84 / 87%) but
covers fewer positions, so the gain shrinks to +0.14 / +0.12 / +0.08. All 14 months (2026-11-13): 12,029 positions with history, **57.30% (+0.209)**, unchanged. One lookup table,
no training: Maia-3 + exact-repeat memory reaches ChessMimic's 57.3. Learned per-player style on new positions
has to add the rest.

**On the real benchmark, with real histories, exact-repeat memory is worth about +0.3 points of top-1 over a
population mode**: 2.6× the 2026-10 one-month estimate and 1.8× its test-mix reweighting. The test oversamples
heavy, strong players. The figure is still a lower bound, because test games from June to December have no history
after May yet (they show the lowest gains), and transpositions are not merged. The crowd is a weaker baseline than
Maia-3. The real headline comparison is own-memory against Maia-3's own predictions on these positions, which needs
the item 9a GPU run (`--save-positions`) joined to data/allie/memory_positions.parquet.

*Opening play predicts middlegame style* (`doppel_lab/opening_link.py`, `lab/results/opening_link.json`,
2026-11-12; 1500-band train table with net evals, 26,757 players with ≥ 16 games, odd / even game halves, rating
regressed out). Opening features from one half (own moves at plies 0–9: Maia log-prob, share off Maia's top move
and off its top 3, entropy faced, net win-chance loss, first-move choice; think time was missing at these plies and
dropped) predict each middlegame trait (plies ≥ 10) in the *other* half by ridge, 5-fold CV over players:

| Middlegame trait | Trait's own other half (ρ) | Opening, other half (ρ) | Ratio | Best single opening feature |
|---|---|---|---|---|
| time allocation | +0.51 | +0.32 | 63% | Maia log-prob +0.27 |
| steering | +0.49 | +0.33 | 68% | off-top-move share +0.24 |
| Maia log-prob | +0.45 | +0.26 | 57% | Maia log-prob +0.25 |
| entropy faced | +0.33 | +0.20 | 60% | Maia log-prob −0.09 |
| blunder rate | +0.24 | +0.15 | 62% | win-chance loss +0.12 |
| opponents' blunder rate | +0.20 | +0.18 | 92% | win-chance loss +0.16 |
| obvious misses | +0.13 | +0.12 | 95% | Maia log-prob −0.11 |

The ratio compares with the trait's own other half, which is not a strict ceiling: averaged opening features can be
less noisy than a middlegame trait, so ratios near or above 100% (sharp-minus-quiet loss, 145%, ρ only +0.09) mean
the trait itself is noisy. **A player's ten opening plies carry about 60% of what their other games' middlegames
say about their middlegame style.** Players who leave Maia's top moves early steer into sharper middlegames and
spread their time differently. Caveat: within-band skill not captured by rating could drive part of the Maia
log-prob and blunder links; steering and allocation are nearly rating-free (§8 sharpness). For the benchmark,
plies 0–9 are scored out but stay legitimate context: the test game's own opening is a cheap per-game z input, and
earlier games' openings already reach the set encoder through the per-game embedding.

*In-game state: is there anything for a per-turn z update to track?* (`doppel_lab/ingame_state.py`,
`lab/results/ingame_state.json`, 2026-11-12; same table, 35.4M non-forced own moves, 26,757 players.) The idea: update
z every turn from the moves already played in this game, plies 0–9 included (the benchmark scores ply k given the
prefix, so this is legal). It can only help if a player's deviation from Maia has a component shared within a game
but varying between that player's games. Each signal was residualised on position, clock and rating covariates
(Maia entropy and top-p, ply bin, clock fraction, legal-move count, both Elos). Variance shares of one move's
residual come from pairwise covariances: two moves from different games of the same player give the player part;
two moves from the same game, minus that, give the game part. "Prefix adds" is the R² gained at a scored move
(ply ≥ 10) by adding the mean of this game's earlier own moves to the player's leave-this-game-out mean.

| Signal (residual) | Player share | Game share | Same-game corr at lag 1 / 5 / 20 own moves | Prefix adds R² (3–7 prior moves / 25+) |
|---|---|---|---|---|
| move surprise (log p + Maia entropy) | 0.62% | −0.20% | −0.004 / +0.001 / −0.002 | +0.0016 / +0.0001 |
| off Maia's top move | 0.29% | −0.13% | −0.002 / −0.001 / −0.002 | +0.0003 / +0.0000 |
| think fraction | 8.1% | 9.1% | +0.26 / +0.13 / +0.03 | +0.056 / +0.081 (last 5 moves: +0.16 more) |
| win-chance loss | 0.25% | 3.4% | +0.19 / +0.06 / +0.00 | +0.022 / +0.022 (last 5: +0.036 more) |

**Move choice has no in-game state.** Once the player is known, this game's earlier deviations from Maia say
nothing about the next one: the game share is slightly negative and every lag is zero. A per-turn z update cannot
move top-1 through move choice, so the benchmark headline must come from per-player z and memory. **Think time and
error size do have in-game state, and it is strong:** think time's game share exceeds its player share, and both
signals decay over about 10–20 own moves, with the last five moves carrying the most (a tempo regime, time
pressure, a lost or messy position). That suits a recurrent think head and blunder model, not the policy. Caveat:
the residualisation is linear in a few covariates, so the loss signal's game part may be position state (a worse
or sharper game) rather than the player's own state (tilt). Either way the prefix is legitimate input. The ±0.2%
negative game shares are small and may be a residualisation artefact.

*v0g_a: a per-player KL gives z reach* (`v0e_train --kl-per-player --w-con 0.1`, 32,808 steps, eval 2026-11-13 on
1500 valrich: 600 held-out players, targets at their history positions 40–47, context = the k most recent earlier
games, so context and target never share a game). The only change from v0f is that each sample's KL is weighted by
1 / (its player's train games), so a player's z pays its KL once per epoch instead of once per game (backlog item 7).

| | v0f | v0g_a |
|---|---|---|
| KL per target game (nats) | 0.86 | 31.8 (19.8 at k = 1 → 41.3 at k = 40) |
| ce_post − ce_teacher (nats/move) | −0.0013 | **−0.049** (−0.063 at k = 40) |
| top-1, posterior z / teacher | 55.61 / 55.65% | **57.18** / 55.65% (57.72% at k = 40) |
| prior-z top-1 / ce − teacher | 55.59% / +0.000 | 55.59% / +0.0006 |
| z-swap gain (nats) [95% CI] | 0.00115 [0.0009, 0.0014] | **0.108** [0.103, 0.114] |
| repeat-visit fidelity, share of history ceiling | 1.2% | **65%** |
| move-choice steering: ρ with held-out half / model SD ÷ real SD | +0.17 / 3% | **+0.47 / 20%** |
| conformity ρ (z adds) | −0.11 (−0.096) | **+0.30 (+0.33)** |
| time allocation, z adds (ply ≥ 10) | +0.06 | +0.125 |
| think r (posterior z) / resign AP | 0.708 / 0.36 | 0.714 / 0.44 |

Every gate in item 7 passes that this eval measures. The steering/conformity coupling and re-ID were not part of this
run. The absolute top1_prior threshold (0.5615) came from a different eval set: v0f itself scores 0.5559 here, and
v0g_a matches it. **One line of loss accounting moved z from 1 nat of style to 30–40 nats: +1.5 points of top-1 and
0.05 nats/move on held-out players, two orders of magnitude over v0f.** Most of it is likely opening repertoire
(repeat-visit share 65%). These targets include plies 0–9, so the benchmark-relevant number (ply ≥ 10, 30 s cut)
is still to measure. The middlegame style metrics, however, all moved too (steering ρ 0.17 → 0.47, conformity flipped
sign).

By ply of the target position (same eval, 2026-11-13, `by_ply`; all k > 0 pooled):

| Plies | Positions | Teacher top-1 | v0g_a top-1 | Δ top-1 | ce − teacher (nats) | z-swap gain (nats) |
|---|---|---|---|---|---|---|
| 0–9 | 40,974 | 51.6% | 59.6% | **+8.0** | −0.249 | 0.56 |
| 10–19 | 40,890 | 53.1% | 53.6% | +0.45 | −0.017 | 0.032 |
| 20+ | 148,920 | 57.4% | 57.5% | +0.03 | −0.0025 | 0.0041 |
| **10+ (benchmark plies)** | 189,810 | 56.5% | 56.6% | **+0.13** | −0.0056 | 0.010 |

**The z's reach is almost all opening repertoire.** Plies 0–9 gain 8 points, and on the plies the benchmark scores the
gain is +0.13 points, most of it at plies 10–19 where repertoire still runs. Past ply 20 z adds 0.03 points and 0.0025
nats. That is still 2–4× v0f's whole-game effect and significant (z-swap CI [0.0031, 0.0050]), but it is generalised
style at the scale of the noise. On benchmark plies, v0g_a's +0.13 is the same order as exact-repeat memory's
+0.21 on the Allie test set.

v0g_b (`--res-lr-mult 0.3`, eval 2026-11-13) is slightly ahead of v0g_a everywhere it was measured: top-1 post
57.26 vs 57.17%, z-swap 0.114 [0.108, 0.120] vs 0.108, repeat share 68 vs 65%, conformity ρ +0.33 vs +0.30;
benchmark plies (10+) +0.16 vs +0.13 points, −0.0063 vs −0.0056 nats. Prior unchanged (55.63%), think r 0.713.

*The Doppel on the Allie test set* (`doppel_lab/allie_v0e.py`, `lab/results/allie_v0e_v0g_a.json`, 2026-11-13).
First test on the benchmark itself, on the 1400–1600 bucket (mean Elo of the two players; 2,021 test games, 4,042
sides, **97,306 protocol positions**, ~11% of the test set). z comes from each player's k most recent earlier 2022
games from the Lichess dumps (game start before the test game, test games never used as context; median 40
available, 60 sides with none). v0g_a was trained on 2026-10 1500-band games and is applied to 2022 unchanged. Every
protocol position is scored. Game bootstrap, 2,000 draws:

| | top-1 | vs Maia-3 (`top1`, 56.63%) [95% CI] |
|---|---|---|
| Maia-3 through the Doppel's trunk (teacher) | 56.55% | −0.07 [−0.14, +0.00] |
| v0g_a, prior z (k = 0) | 56.58% | |
| v0g_a, k = 1 / 5 / 10 / 40 | 56.64 / 56.73 / 56.71 / **56.75%** | **+0.12 [+0.01, +0.23]** (k = 40) |
| Maia-3 + exact-repeat memory | 56.73% | +0.10 |
| **v0g_a + exact-repeat memory** | **56.84%** | **+0.21 [+0.10, +0.31]** |

Against its own teacher on identical inputs, v0g_a gains +0.20 [+0.09, +0.31] points and −0.006 nats/move (1.3171 →
1.3111). The gain is +0.60 points at plies 10–19 and +0.09 from ply 20 on. Memory and z stack: v0g_a + memory beats
Maia-3 + memory by +0.11 [+0.00, +0.22]. **The first per-player win on the benchmark's own positions: a Doppel
trained on 2026 data, given a player's real earlier games, beats Maia-3 on the 2022 test games, and the gain grows
with context (k = 1 → 40).** It is small, about the size of the memory switch. The teacher path reads Maia-3 slightly
differently from `top1.py` (−0.07): same model and ratings, but a different board-history encoding, still to
reconcile. Gains are against `top1.py` to stay conservative.

v0g_b on the same bucket (`lab/results/allie_v0e_v0g_b.json`; same bootstrap, positions aligned with v0g_a):

| | v0g_a | v0g_b | v0g_b [95% CI] |
|---|---|---|---|
| Doppel (k = 40) vs Maia-3 | +0.12 | +0.19 | [+0.08, +0.29] |
| Doppel + memory vs Maia-3 | +0.21 | **+0.26** | [+0.16, +0.37] |
| Doppel + memory vs Maia-3 + memory | +0.11 | +0.16 | [+0.06, +0.26] |
| top-1, k = 0 / 1 / 5 / 10 / 40 | | 56.57 / 56.62 / 56.78 / 56.73 / 56.81% | |

v0g_b minus v0g_a is +0.06 [−0.02, +0.14], so the two are not separable here. v0g_b is ahead on every measure,
though, as it was on valrich. With v0g_b, z beats memory alone outright: the CI on "+ memory vs Maia-3 + memory"
now excludes zero. Next: v0g-style arms at 1100 and 1900 (`--band`, v0g_b flags; training started 2026-11-13). They
cover the 1000–2000 buckets, about 65% of the test set. The −0.07 teacher gap is precision, not encoding
(CPU check on 4,609 positions from 80 games where the two disagree, 2026-11-13): both paths feed Maia-3 identical
board tokens and Elo, but the teacher ran under bf16 autocast, which changes the argmax on ~2% of positions vs
fp32. `top1.py`'s fp16 agrees with fp32 on 99.7%. The flips run both ways, so the net −0.07 is noise. `allie_v0e
eval` now runs the trunk in fp16. The 1100/1900 evals use that; the 1500 numbers above are still bf16.

**All three bands, fp16 (2026-11-14).** v0g_b flags per band; 1500 re-evaluated in fp16 (bf16 copy kept at
`data/allie/v0e_positions_v0g_b.bf16.parquet`). Top-1 points vs Maia-3 run in the same pass, game bootstrap 95% CIs
(`lab/results/allie_v0e_v0g_{1100,b,1900}.json`):

| Band (bucket) | Positions | Maia-3 | Doppel vs Maia-3 | + memory vs Maia-3 | Beyond memory | Nats saved/move |
|---|---|---|---|---|---|---|
| 1100 (1000–1400) | 184,599 | 54.40% | **+0.26** [+0.18, +0.35] | +0.30 [+0.21, +0.38] | +0.25 [+0.16, +0.33] | 0.009 |
| 1500 (1400–1600) | 97,306 | 56.63% | **+0.17** [+0.07, +0.27] | +0.25 [+0.15, +0.36] | +0.16 [+0.05, +0.26] | 0.007 |
| 1900 (1600–2000) | 213,192 | 57.80% | **+0.10** [+0.05, +0.16] | +0.31 [+0.24, +0.37] | +0.09 [+0.03, +0.15] | 0.003 |

Every band clears zero on every measure, on 495k positions (56% of the test set). The two levers move in opposite
directions with rating: z's own gain falls from +0.26 (1100) to +0.10 (1900), while exact-repeat memory adds +0.03 at
1100 and +0.20 at 1900 (repeat share 0.5% vs 1.3% of positions). Weaker players are less predictable from the
population and more from their own past play in new positions; stronger players gain from repeating their book.

**Where the gain lives: position difficulty (1500, fp16).** Quintiles of Maia-3's policy entropy over legal moves
(q1 = one obvious move; a no-engine cousin of Lc0-policy "ease", jk_182 2026-11-12, §10). Points vs Maia-3, game
bootstrap (`lab/results/allie_difficulty.json`):

| Entropy quintile | Maia-3 top-1 | Doppel gain | Nats saved/move |
|---|---|---|---|
| q1 (0–0.69) | 91.3% | −0.01 [−0.03, +0.01] | 0.0002 |
| q2 (0.69–1.18) | 67.2% | −0.04 [−0.21, +0.12] | 0.0015 |
| q3 (1.18–1.57) | 53.7% | +0.19 [−0.05, +0.42] | 0.006 |
| q4 (1.57–1.94) | 41.4% | +0.28 [+0.01, +0.55] | 0.013 |
| q5 (1.94–3.26) | 29.6% | **+0.44** [+0.13, +0.73] | 0.013 |

The whole gain is in positions where the population is split; where one move is obvious, a player's history adds
nothing. Personalization is a tie-breaker among plausible moves, so a benchmark over hard positions would show it
larger.

**Retrieval step 0: kNN over a player's own positions, no training (2026-11-14, `lab/doppel_lab/allie_knn.py`,
`lab/results/allie_knn.json`).** Keys are Maia-3's pooled final-layer state (mean over squares, PCA to 128, unit
norm). The store is the mover's positions in their 40 most recent earlier games (median 1,298 positions). Up to 32
neighbours vote for their played move with weight exp((cos−1)/τ), the vote is mixed into Maia-3, and the mix is used
only when the best neighbour clears a threshold. n, τ, λ and the threshold are tuned on one game-hash half and
scored on the other. 1500 bucket, 97,306 positions:

| Store | Top-1 vs Maia-3 |
|---|---|
| The player's own games | **+0.11** [+0.07, +0.15] |
| Shuffled players (control) | −0.00 [−0.003, 0.00] |
| (trained v0g_b, same positions) | +0.17 |

The signal is the player's: the control is flat. It is +3.5 on the 805 exact repeats and +0.08 elsewhere. By ply it
is +0.55 at 10–19 and 0.00 from ply 20 on. By best-neighbour cosine it is 0.00 below 0.9 (98% of positions), then
+3.7 / +7.3 / +8.3 / +10.6 at 0.90 / 0.95 / 0.98 / 0.995. Frozen Maia-3 states find near-transpositions of the
player's book and nothing more, a soft repertoire memory. Generalizing to new positions needs learned keys or a
learned residual (step 1 of the retrieval plan), which v0g's z already partly does.

**Full test set (2026-11-15).** The outer buckets have no in-band model, so they borrow the nearest one
(`MODEL_BAND` in `allie_v0e.py`; out-of-band rows in `data/allie/v0e_positions_<run>_at<band>.parquet`):

| Bucket | Model | Positions | Maia-3 | Doppel vs Maia-3 | Beyond memory |
|---|---|---|---|---|---|
| < 1000 | v0g_1100 | 108,121 | 50.53% | +0.23 [+0.12, +0.34] | +0.23 [+0.12, +0.34] |
| ≥ 2000 | v0g_1900 | 280,831 | 61.01% | **−0.09** [−0.15, −0.04] | −0.10 [−0.15, −0.05] |

All five buckets together are the whole test set (884,049 positions). Maia-3 gets 57.09% (published 57.1%).
Doppel gets 57.19% (+0.10) and Doppel + memory gets **57.39% (+0.30)**; Maia-3 + memory gets 57.30%. The 1100 model
carries down to < 1000 almost unchanged; the 1900 model hurts above 2000, where memory alone is worth +0.33
(repeat share 2.6%). Dropping z above 2000 (Maia-3 + memory there) would give 57.42%, but that rule was chosen
after seeing the result, so it is not a headline number. An in-band ≥ 2000 model is the fix.

Paired tests on the same 884,049 positions (18,138 games), against Maia-3 run in the same pass (McNemar with
continuity correction; game-cluster bootstrap, which respects within-game correlation and is the one to quote):

| Comparison | Gain | Discordant (win / loss) | McNemar p | Game bootstrap 95% |
|---|---|---|---|---|
| Doppel vs Maia-3 | +0.10 | 11,260 / 10,401 | 6e-9 | [+0.06, +0.13] |
| Doppel + memory vs Maia-3 | +0.30 | 13,600 / 10,979 | 1e-62 | [+0.26, +0.33] |
| Doppel + memory vs Maia-3 + memory | +0.09 | 10,954 / 10,176 | 9e-8 | [+0.06, +0.12] |

Leakage audit (2026-11-15): context and memory use only games with start time strictly before the test game's
`created_at` (`allie_v0e.select`, `allie_memory`), and every Allie test game is excluded from all history, not just
the same game. The models were trained on 2026 games and the test is 2022. Recipe choices were made on 2026 valrich
data; MODEL_BAND was set before the outer buckets were scored. This is a separate protocol, **Allie test with prior
player history**: Maia-3 and Allie never see the player's earlier games, so it is not a like-for-like leaderboard
entry. Two-thirds of the +0.30 is exact-repeat memory, which needs no model.

**Retrieval step 1: learned lookup keys (2026-11-15, `lab/doppel_lab/retr_train.py`, `lab/results/allie_retr1.json`).**
Same inputs as step 0 (frozen Maia-3 pooled states, PCA-128; store = the mover's 40 most recent earlier games). A
small head learns query and key projections (128 → 64), a temperature, and a gate on best similarity and ply; it
turns the attention-weighted vote over stored played moves into a logit residual on Maia-3. Trained on 2,000 2026
1500-band players (store 40 games, queries in their last 10), epoch chosen on valrich (+0.45 there); nothing tuned on
Allie. 1500 bucket, 97,306 positions, game bootstrap:

| | Top-1 vs Maia-3 |
|---|---|
| **Learned lookup, own store** | **+0.34** [+0.22, +0.45] |
| Same, on exact repeats (805) / elsewhere | +8.1 / +0.27 |
| Shuffled-player store (control) | −0.33 [−0.44, −0.23] |
| (frozen kNN, step 0) | +0.11 |
| (v0g_b z; v0g_b + memory) | +0.17; +0.25 |

The learned lookup beats both the frozen kNN and v0g on the same positions, with no trunk or z. It still lives in the
opening (+1.47 at plies 10–19, +0.28 at 20–29, −0.08 at 30–39, 0.00 after), but it now reaches further from exact
matches: +1.12 at best cosine 0.8–0.9, where step 0 had nothing. The control is strongly negative: the head trusts
the store, so a wrong or unrepresentative store hurts. Next: combine with z (and memory), and train a ≥ 2000 band.

**Learned lookup in every bucket (2026-11-16, `lab/results/allie_retr1_<band>.json`).** Same recipe, one head per
training band (the < 1000 bucket borrows the 1100 head); epoch chosen on valrich; game bootstrap; shuffled-player
store as the control.

| Bucket | Positions | Lookup − Maia-3 | Shuffled store | Doppel (v0g) − Maia-3 |
|---|---|---|---|---|
| < 1000 (1100 head) | 108,121 | +0.23 [+0.14, +0.33] | −0.24 | +0.23 (1100 model) |
| 1000–1400 | 184,599 | +0.28 [+0.21, +0.36] | −0.25 | +0.26 |
| 1400–1600 | 97,306 | +0.34 [+0.22, +0.45] | −0.33 | +0.17 |
| 1600–2000 | 213,192 | +0.46 [+0.39, +0.53] | −0.16 | +0.10 |
| 2000+ | 280,831 | +0.38 [+0.32, +0.43] | −0.11 | +0.00 (in-band 2300 model) |
| **All (884,049)** | | **+0.36 [+0.32, +0.39]** | −0.19 | +0.10 |

On the whole test set the lookup gets **57.44% top-1 against Maia-3's 57.08%** (McNemar χ² = 419; 13,341 vs
10,200 discordant), above Doppel + memory (57.39%) with no trunk change and no z. Up to 2000 the lookup grows with
rating while the z model shrinks: stronger players repeat their own lines more (exact-repeat positions: 125, 875,
805, 2,811 and 7,413 across the buckets; gain on them +2.4 to +8.1). Both halves of each bucket agree within 0.12.
The in-band 2000+ z model (v0g recipe on a 2000+ pool, `allie_v0e_v0g_2300.json`) removes the −0.09 the 1900 model
cost there but adds nothing (+0.00; + memory +0.42, the same as Maia-3 + memory).
The gain is in the opening at every band (plies 10–19: +0.8, +1.2, +1.5, +2.1) and rises with best cosine (1900: +1.2 at 0.8–0.9, +5.6 at 0.9–0.95, +16 above 0.995).

**Combo: z and the learned lookup together (2026-11-17, `doppel_lab/combo.py`, `lab/results/allie_combo_full.json`,
`data/allie/combo/combo_fit_<band>.json`).** Both are corrections to frozen Maia-3's log-probs: zres = v0g Doppel −
Maia-3 (z from 40 earlier games), rlook = lookup head − Maia-3. Combo = Maia-3 + a·zres + b·rlook, (a, b) per ply
bin on a 0–2 grid, chosen on valrich (600 held-out 2026 players per band, games 40–47; 96–97% of the lookup's valrich
queries have a z residual, the rest are players with > 48 valrich sides whose targets differ between the two
pipelines). < 1000 borrows the 1100 weights. Check: Maia-3 + mapped zres reproduces the Doppel's argmax on every
smoke row, and the lookup and Doppel arms reproduce `allie_retr1_full` and v0g per bucket.

| Valrich (held-out, top-1 pts vs Maia-3) | 1100 | 1500 | 1900 | 2300 |
|---|---|---|---|---|
| Lookup | +0.36 | +0.45 | +0.46 | +0.39 |
| Doppel (z) | +0.31 | +0.21 | +0.10 | +0.03 |
| Sum (a = b = 1) | +0.58 | +0.51 | +0.55 | +0.39 |
| Combo, weights from the other half of players | +0.49 | +0.56 | +0.49 | +0.38 |

Whole Allie test set (884,049): **combo 57.53%** vs lookup 57.44% vs Maia-3 57.08%: **+0.44 [+0.40, +0.48]** over
Maia-3, **+0.08 [+0.05, +0.11]** over the lookup (McNemar χ² = 32). The plain sum gets the same (57.52%), so the
grid adds nothing beyond "add both". Over the lookup by bucket: +0.19, +0.16, +0.11, +0.06, −0.01 (< 1000 … 2000+), the
z model's own pattern. By ply (combo − Maia-3): +1.71, +0.28, +0.11, +0.04; the lookup alone: +1.54, +0.20, 0.00,
+0.01. So z adds a little at every ply and is the only part with anything after ply 30 (+0.09 there on its own).
Control (real z, shuffled-player lookup store): −0.06.

**Lookup store size (2026-11-17, `doppel_lab/lookup_k.py`, `lab/results/lookup_k.json`).** Same head, store = the
side's k most recent earlier games, same positions.

| k | 1 | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| 1400–1600 | +0.03 | +0.08 | +0.15 [0.09, 0.21] | +0.26 | +0.30 | +0.34 |
| 1600–2000 | +0.05 | +0.09 | +0.16 [0.11, 0.20] | +0.26 | +0.36 | +0.46 |

Not saturated at 40, most clearly at 1600–2000 (stronger players repeat lines more). About 8% of sides have fewer
than 40 earlier games; the k = 40 gain on sides with a full 40 is the same within 0.01.

Store to 200 games, 1400–1600 only (`doppel_lab/lookup_store.py`, `lab/results/lookup_store.json`): the 200 most recent
earlier 2022 games per side (695,888 games, 46.3M positions; 80% of sides have the full 200), same head, same PCA,
97,306 positions.

| k | 10 | 20 | 40 | 80 | 120 | 200 |
|---|---|---|---|---|---|---|
| gain | +0.26 | +0.30 | +0.34 [0.21, 0.45] | **+0.48** [0.35, 0.61] | +0.49 | +0.49 [0.33, 0.63] |

The curve keeps rising to 80 games and is flat after. 40 → 80 adds +0.14. The head never saw stores this large in
training, so the plateau may belong to the head as much as to the data.

**Ceiling probe: a per-player adapter inside Maia-3 (2026-11-17, `doppel_lab/lora_probe.py`,
`lab/results/lora_probe.json`).** Question: is there personal signal at plies 20+ that an in-trunk method could reach,
and how does it grow with games? 199 held-out 2026-10 players with ≥ 545 games in one band (77 at 1500, 122 at 1900):
train on the N games before 5 val games, test on the last 20 (129,890 positions at ply ≥ 10). Rank-8 LoRA on the MLP
(linear1/linear2) of blocks 5–8, one per player, trained together; AdamW lr 1e-3, wd 0.01; epoch per player on val CE
(epoch 0 = Maia-3). Control: each player's test through another player's adapter (same band). Game bootstrap.

| N games | All | 10–19 | 20–29 | 30–39 | 40+ | Control (all) |
|---|---|---|---|---|---|---|
| 40 | **+0.79** [0.61, 0.96] | +4.10 | +1.26 [0.81, 1.74] | +0.28 [−0.16, 0.71] | −0.13 | −1.90 |
| 200 | **+0.89** [0.70, 1.08] | +5.91 | +1.77 [1.27, 2.29] | +0.31 [−0.18, 0.83] | −0.59 | −2.94 |
| 500 | **+0.79** [0.57, 1.01] | +6.87 | +1.53 [0.97, 2.10] | +0.14 [−0.39, 0.66] | −0.93 | −3.69 |

By band (N = 40 / 200 / 500): 1500 +1.20 / +1.58 / +1.54; 1900 +0.54 / +0.48 / +0.35. Best epoch is mostly 1–2.

Reading. (1) Plies 20–29 carry real personal signal reachable in the trunk (+1.3 to +1.8), several times the lookup's
+0.20 there on Allie. (2) Plies 30–39: nothing distinguishable from zero at any N, so the brainstorm's gate ("N = 500
gives < +0.2 at 30+, CI touching 0") says middlegame personalisation beyond ply 30 is out of reach for this kind of
adapter even with 500 games. (3) More games buy opening gains and cost the endgame (40+ goes −0.13 → −0.93): the
adapter learns the player's openings and drifts elsewhere, and per-player early stopping on overall CE can't see it.
A ply gate (adapter off at ply ≥ 40) is the obvious fix and is fixed here, from these players, before any Allie use.
Caveat: these are very heavy players (a game every ~1.5 h for a month), more repetitive than typical test sides.

Lookup on the same players (`doppel_lab/probe_lookup.py`, `lab/results/probe_lookup.json`): identical 129,890 test
positions; store = the same N training games; retr_train's head for the player's band, unchanged. "Adapter, gated" =
the probe adapter with Maia-3 at ply ≥ 40, computed from the per-ply rows above.

| N | Lookup all | Lookup 10–19 | Lookup 20–29 | Lookup 30+ | Adapter all | Adapter, gated |
|---|---|---|---|---|---|---|
| 40 | +0.51 [0.42, 0.62] | +2.67 | +0.44 [0.16, 0.74] | ≈ 0 | +0.79 | +0.86 |
| 200 | +0.64 [0.51, 0.76] | +3.89 | +0.52 [0.13, 0.92] | ≈ 0 | +0.89 | +1.22 |
| 500 | +0.65 [0.51, 0.79] | +4.08 | +0.39 [−0.09, 0.81] | ≈ 0 | +0.79 | +1.30 |

The adapter beats the lookup in every ply bin below 30: 10–19 by ~1.5–2.8, 20–29 by ~3×. The lookup never hurts
at ply 40+ (CI within ±0.15), and the ungated adapter does. With the gate, the adapter is about 2× the lookup and keeps
improving with N, while the lookup flattens after ~200 games (the head was trained on 40-game stores). By band, the
lookup gives 1500 +0.57/+0.77/+0.78 and 1900 +0.48/+0.56/+0.57. At 1900 the lookup matches or beats the ungated
adapter (+0.54/+0.48/+0.35).

**Test-time adapters on the Allie test set (2026-11-17, `doppel_lab/allie_ttt.py`, `lab/results/allie_ttt_1500.json`,
`allie_stack_1500.json`).** The probe recipe on real test sides, 1400–1600 bucket only so far. Context = the side's
40 most recent earlier games (the same games z and the lookup use). Val = the 5 most recent, train = the other 35.
One rank-8 LoRA per side, sides trained 200 at a time, epoch per side on val CE (epoch 0 = Maia-3; picked: 0 for 9%,
1 for 68%, 2 for 22%). Scored on the bucket's 97,306 protocol positions. Control: each side's test through another
side's adapter. Game bootstrap. About 48 min of GPU time for 4,008 sides.

| | All | 10–19 | 20–29 | 30–39 | 40+ |
|---|---|---|---|---|---|
| Adapter | **+0.80** [0.60, 0.99] | +2.59 | +0.90 [0.49, 1.36] | +0.67 [0.24, 1.15] | −0.03 [−0.30, 0.26] |
| Control | −1.78 [−1.95, −1.59] | −2.53 | −1.64 | −1.58 | −1.57 |

Stack, equal weights, no fitting (Maia-3 log-probs + each residual), same positions. "ttt40" = adapter off at
ply ≥ 40 (the gate fixed from the probe players). Last column is paired against ttt40 alone.

| Arm | Gain | vs ttt40 |
|---|---|---|
| z | +0.17 [0.06, 0.26] | −0.64 |
| lookup | +0.34 [0.22, 0.45] | −0.47 |
| z + lookup | +0.48 [0.34, 0.63] | −0.33 [−0.50, −0.16] |
| ttt40 | **+0.81** [0.67, 0.96] | 0 |
| lookup + ttt40 | **+0.83** [0.67, 1.01] | +0.02 [−0.09, 0.12] |
| z + ttt40 | +0.74 | −0.07 [−0.16, 0.02] |
| z + lookup + ttt40 | +0.66 | −0.15 [−0.26, −0.02] |

Reading. (1) Training a small adapter on the side's 35 games at test time gives 1.7× z + lookup on the same games,
and it is the first method with a clear gain at plies 20–39 on the Allie protocol. The probe players showed ≈ 0 at
30–39; the difference is unexplained (the Allie protocol stops before the first move under 30 s, the probe does not).
(2) The adapter subsumes the other two: adding the lookup is +0.02, n.s., and adding z at equal weight hurts. It is the
same history counted twice, so equal weights over-correct. Fitted weights would need a fitting set with adapters
trained on it (valrich), which has not been run. (3) The ply-40 gate changes little here (+0.01). At 40 games the
adapter barely moves the endgame. That matches the probe, where the endgame loss grew with N.

All buckets (`allie_stack_{700,1100,1900,2300}.json`, `allie_ttt_1500_ctx200.json`; full set
`lab/results/allie_stack_full.json`, `combo stack-full`).

| Bucket | Positions | z + lookup | ttt40 | lookup + ttt40 | lookup + ttt40 vs ttt40 |
|---|---|---|---|---|---|
| 700 | 108,121 | +0.35 | **+0.89** | +0.84 | −0.05 |
| 1100 | 184,599 | +0.44 [0.34, 0.53] | **+0.91** [0.79, 1.01] | +0.87 | −0.04 [−0.12, 0.04] |
| 1500 | 97,306 | +0.48 [0.34, 0.63] | +0.81 [0.67, 0.96] | +0.83 | +0.02 [−0.09, 0.12] |
| 1900 | 213,192 | +0.53 [0.44, 0.61] | +0.65 [0.56, 0.75] | **+0.81** [0.70, 0.92] | +0.16 [0.09, 0.23] |
| 2300 | 280,831 | +0.36 | +0.30 | **+0.55** | +0.25 |

**Full Allie test set (884,049 positions, every one covered; Maia-3 57.09%), equal weights, no fitting on test:**

| Arm | Top-1 | vs Maia-3 | vs z + lookup |
|---|---|---|---|
| z + lookup (the combo) | 57.52% | +0.43 [0.38, 0.47] | 0 |
| ttt40 | 57.73% | +0.64 [0.59, 0.69] | +0.21 [0.16, 0.26] |
| **lookup + ttt40** | **57.83%** | **+0.75** [0.69, 0.79] | **+0.32** [0.27, 0.36] |
| z + lookup + ttt40 | 57.76% | +0.67 | +0.24 |

By ply (lookup + ttt40 vs Maia-3): +2.68 / +0.74 / +0.34 / +0.01. The ply-40 gate was fixed from the probe players
before any Allie use; the arm set (all subsets, gated and not) was fixed before the first stack ran. 57.83% is the
best of the 11 arms, so a small selection effect applies, but the runner-up gap to the combo (+0.24) is far outside it.
The adapter's edge shrinks with rating (+0.89 at 700 → +0.30 at 2300) while the lookup's contribution on top grows
(−0.05 → +0.25). Stronger players repeat lines verbatim, which the lookup copies and a rank-8 adapter on 35 games only
partly learns. The adapter's GPU cost: about 0.7 s per side (36,223 sides, 7.2 h in all).

One player at a time (`doppel_lab/ttt_timing.py`, `lab/results/ttt_timing_{cuda,cpu}.json`): 5 Allie 1500 sides
with a full 40-game context (~870–1,130 train positions), fresh adapter each, after one warm-up side. RTX 5090, bf16:
median **0.69 s** train + 0.28 s board encoding, peak 4.7 GB VRAM. CPU, 16 threads, fp32: median **80 s** (74–95 s).
Model load is under 1 s either way. Batching 200 sides buys nothing per side: one at a time is already 0.7 s.

More context: 1500 with 200 context games (195 train, same val rule; 3.8 h of GPU):
adapter **+1.01** [0.80, 1.22] vs +0.80 at 40, by ply +3.59 / +1.12 / +0.22 / +0.07. More history helps the opening
most, and the 30–39 gain seen at 40 games (+0.67) does not appear.

**200 context games, all buckets (2026-11-18; `allie_ttt_<band>_ctx200.json`, `allie_stack_<band>.json`,
`allie_stack_full.json`).** Same recipe, the 200 most recent earlier games (`lookup_store.py select --band`,
`allie_ttt run --ctx 200`); sides with fewer use what they have (≥ 200 available: 700 44%, 1100 69%, 1900 83%, 2300
79%). Stack arms ttt200 (ungated) and ttt200g40 (off at ply ≥ 40, the same fixed gate) were added to the 11; 19 arms
in all, the gate and arm list fixed before the run. About 32 h of GPU over the five buckets (2300 alone 11.2 h).

| Bucket | Maia-3 | ttt40 (40 games) | ttt200 | **ttt200g40** | ttt200 by ply (10–19 / 20–29 / 30–39 / 40+) | Control |
|---|---|---|---|---|---|---|
| 700 | 50.5% | +0.89 | **+1.28** [1.07, 1.46] | +1.03 | +2.57 / +1.50 / +0.53 / +0.66 | −2.41 |
| 1100 | 54.4% | +0.91 | **+1.18** [1.03, 1.32] | +1.11 | +3.34 / +1.25 / +0.88 / +0.16 | −2.52 |
| 1500 | 56.6% | +0.81 | **+1.01** [0.80, 1.22] | +0.98 | +3.59 / +1.12 / +0.22 / +0.07 | −2.70 |
| 1900 | 57.8% | +0.65 | +0.93 [0.78, 1.07] | **+1.02** | +4.05 / +0.98 / +0.35 / −0.21 | −2.36 |
| 2300 | 61.0% | +0.30 | +0.04 [−0.08, 0.15] | **+0.48** | +2.87 / +0.20 / −0.55 / −0.96 | −1.98 |

Control = the same positions scored with another side's 200-game adapter (ungated).

**Full test set (884,049 positions; Maia-3 57.09%):**

| Arm | Top-1 | vs Maia-3 | vs z + lookup |
|---|---|---|---|
| lookup + ttt40 (previous best) | 57.83% | +0.75 [0.69, 0.79] | +0.32 [0.27, 0.36] |
| ttt200 | 57.84% | +0.75 [0.68, 0.82] | +0.32 [0.25, 0.39] |
| **ttt200g40** | **57.95%** | **+0.86** [0.81, 0.92] | **+0.43** [0.38, 0.49] |
| lookup + ttt200g40 | 57.97% | +0.88 [0.82, 0.94] | +0.45 [0.39, 0.50] |

Paired, game bootstrap: ttt200g40 − ttt40 **+0.23** [0.18, 0.28]; ttt200g40 − lookup + ttt40 +0.12 [0.06, 0.18]; the
gate at 200 games +0.11 [0.07, 0.16]; adding the lookup to ttt200g40 +0.015 [−0.02, 0.05], n.s. So the headline is one
method, the gated 200-game adapter, by ply +3.28 / +0.89 / +0.18 / 0. At 200 games the adapter absorbs what the lookup
added at 40 (the repeated lines).

Readings. (1) More history helps every bucket's opening (10–19: +2.6 to +4.1, the largest at 1900), and helps less or
hurts later. At 2300 the ungated adapter gains +2.87 in the opening and loses −0.96 at 40+, netting zero; gated it is
+0.48. The fall of the ungated gain with rating is therefore mostly a late-game cost, not less personal play overall.
(2) Maia-3's own top-1 at 2300 is 60.4% at 10–19 and 62.5% at 40+, so a strong base model sets the size of the late
headroom but does not explain a negative sign: an adapter with nothing to learn should land near 0 after epoch
selection. The likelier cause is that the epoch is chosen on validation CE over all plies, which the opening dominates;
the fit spills into later positions where there is little personal signal. Per-phase epoch choice is untested.
(3) The swap control stays at −2.0 to −2.7 at every band, while the own gain shrinks at 2300: players are no more
interchangeable at 2300, but less of their difference is reachable beyond the opening. Headroom-normalised ttt200
gains (gain / (1 − Maia top-1)): 2.58 / 2.59 / 2.33 / 2.19 / 0.10% of Maia's misses at 700–2300.
(4) 19 arms were compared; the winner over the runner-up (+0.015) is within selection noise, the gated adapter's gain
over the 40-game stack (+0.12) is not.

Rating drift and staleness (`doppel_lab/elo_drift.py`, `lab/results/elo_drift_1500.json`), same bucket, same hits.
Drift = the side's Elo in the test game minus the median of its own Elo over its 40 context games. Span = days from
the oldest context game to the test game. 59 sides with no context (1,185 positions) are in no bin. Side drift is
small: 5–95% range −46 to +49.

| | Positions | z + lookup | Adapter (ttt40) |
|---|---|---|---|
| \|drift\| < 40 | 83,170 | +0.54 [0.38, 0.68] | +0.85 [0.69, 1.01] |
| \|drift\| 40–100 | 11,353 | +0.22 [−0.16, 0.62] | +0.74 [0.32, 1.12] |
| \|drift\| ≥ 100 | 1,598 | +0.06 [−0.70, 0.88] | −0.31 [−0.91, 0.29] |
| span < 7 days | 56,486 | +0.61 [0.44, 0.80] | +0.99 [0.79, 1.19] |
| span 7–30 | 27,051 | +0.37 [0.11, 0.62] | +0.71 [0.44, 0.99] |
| span 30–90 | 7,260 | +0.28 [−0.18, 0.76] | +0.61 [0.07, 1.12] |
| span ≥ 90 | 5,324 | +0.09 [−0.39, 0.61] | −0.15 [−0.64, 0.37] |

Both methods fade as the history gets older, and the adapter at 90+ days is indistinguishable from zero. Moderate drift
(40–100) costs the adapter little. Large drift is too rare here (1.7%) to read. The span is confounded with how often
the player plays: 40 games in a week is a heavier, likely more repetitive player. So this does not yet show that old
games are stale. Separating the two needs recency weighting within a player, or the same player at two spans.

**Revisits: the same player at the same board (2026-11-15/16, `doppel_lab/revisits.py`,
`doppel_lab/revisit_model.py`, `lab/results/revisits_*.json`, `revisit_model.json`, `revisit_learn.json`,
`revisit_triples.json`).** Held-out games (12.7M); a pair = two consecutive visits by one player to one board
(128-bit Zobrist, either path), scored at ply ≥ 10. Chance = Σp² agreement of Maia-3 at the two visits (real Elos);
personal repetition = observed repeat − chance.

| Ply | Pairs | Repeat | Chance | Personal |
|---|---|---|---|---|
| 10–19 | 1,448,308 | 75.1% | 43.6% | +31.5 |
| 20–29 | 48,734 | 74.4% | 46.9% | +27.5 |
| 30–39 | 1,648 | 75.5% | 48.7% | +26.8 |
| 40+ | 1,310 | 75.9% | 66.3% | +9.6 |

Repetition stays at 75% to ply 39 while chance rises; ply 40+ is mostly endgames reached by other routes, where
the population catches up. Rarity of the board elsewhere in the corpus barely matters (−0.1 to −1.0 pts); the
number of earlier visits in the career does (k = 6+: +9.7 pts vs k = 2). Deep middlegame coincidences are rare:
about 1 in 1,000 pairs are at plies 30–39.

Mixed-effects logistic, logit(chance) offset, random intercept per player (1.32M different-opponent pairs at plies
10–29, 139k players, σ = 0.54): a clock that differs from the first visit costs 2–4 pts; a gap of 100+ games 4.9
pts; a different route 1.7; losing the first game 1.6; the player's own rating change between visits 0. Habits
fade with time, not with improvement. Power: 2-pt effects need about 14k pairs; the deep bins can only resolve
about 5 pts.

Move quality (Stockfish 20k nodes). In 50k pairs, a first-visit blunder is repeated 35.5% of the time against 37.1%
chance: the personal component vanishes (−1.6 [−6.4, +3.2]; good moves +34.0, mistakes +21.7). Inside bins of
Maia-3's probability for the move, blunders still lose 14–36 pts against equally unlikely good moves. Triples
(1.38M chains of three visits, 81k players) split slip from learning:

| Middle visit's move | Also played at the first visit | Kept at the third visit |
|---|---|---|
| Good | yes (1,009,299) | 90.6% |
| Mistake | yes (20,505) | 80.4% |
| Blunder | yes (3,868) | 75.8% [74.1, 77.3] |
| Good | no (220,059) | 44.4% |
| Blunder | no (4,334) | 21.3% [20.1, 22.5] |

Mostly slips: a one-off blunder is dropped four times in five. Blunder habits stick (76% per visit), with some
learning that scales with the error (habitual moves kept at 91% good, 83% inaccuracy, 80% mistake, 76% blunder;
adjusted −13.7 pts for a habitual blunder vs a habitual good move) and a small punishment effect (a lost game B
lowers keeping a habitual blunder by 5.7 pts [3.0, 8.5], a habitual good move by 1.2). Two visits with the same
blunder can coincide without a habit, so the learning share is an upper bound. When a move is changed, the new
move is better and worse equally often (27% each); after a recent blunder it leans worse (worse − better +6.8
[+0.7, +12.9], vs −0.5 otherwise), suggestive tilt from one of five subgroups looked at.

**Coaching gates: simulation fidelity and forecast (2026-11-15/16, `doppel_lab/sim_fidelity.py`,
`lab/results/sim_fidelity.json`, `sim_forecast.json`).** 200 held-out 1500-band players; 40 simulated games each
against pool Doppels (no clock; resignation from the eval); target = their real games 41–100. Per-player correlation
with the real blunder rate (≥ 0.2 per move): first 40 real games 0.56, Doppel at T = 0.8 0.25, Doppel at T = 1 0.14,
plain Maia-3 at T = 1 0.17. Simulated games are 14–18 plies short. (The rating-average baseline reads −0.42: the
players span only 1481–1632, so the leave-one-out ±100 Elo mean is nearly "everyone but you". It is dropped.)

Forecast, out of fold, MSE weighted by moves, as a share of the variance not due to binomial noise:

| Predictor | Share captured | vs shrink-to-mean |
|---|---|---|
| First 40 games, raw | 11% | worse |
| Pooled mean | −2% | worse |
| **First 40 shrunk to the mean (empirical Bayes)** | **36%** | — |
| Shrunk to a Doppel-sim prior (T = 0.8) | 37% | [−0.08, +0.04] ×10⁻⁴ |
| Shrunk to a Maia-sim prior | 36% | tie |
| Linear blend of first 40 and Doppel sim | 37% | tie |

The prior is worth about 670 moves (≈ 20 games). Simulation adds nothing measurable, and Maia does as well as the
Doppel, so even the sliver is not personal. The same holds at ≥ 0.3. For the n = 1 study, weaknesses are ranked from
real-game counts shrunk toward the crowd (PROTOCOL_N1 deviation log, 2026-11-16).

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
- **Five position-complexity measures compared.** jk_182, *Different Ways to Measure the Complexity of Chess
  Positions*, Lichess blog, 2026-11-12, [post](https://lichess.org/@/jk_182/blog/different-ways-to-measure-the-complexity-of-chess-positions/6wL7UC7Y).
  On ~3,000 Olympiad positions (2500+, moves 20–40), "inverted ease" (Lc0 policy-weighted expected-score loss) best
  predicts human mistakes; Biswas–Regan depth difficulty, Lc0 WDL sharpness, WDL sensitivity and Stockfish
  depth-deviation do worse. Ease is move-based (0 with one legal move), so it misreads openings; the author prefers
  WDL sensitivity as position complexity. Our entropy split in §8 is the no-engine analogue. Read in full.

### Clock and tilt

- **Do mistakes provoke new mistakes?** IEEE Transactions on Games ([PDF](https://www.nes.ru/files/faculty/Do-mistakes-provoke-new-mistakes-Evidence.pdf)).
  Tests within-game tilt on tournament data; the counterpart to our shock analysis (§5). Abstract.
- **Mistake rate by clock.** jk_182, [How does the clock impact the rate of mistakes?](https://chessenginelab.substack.com/p/mistakes-clock);
  a 68M-game study of the Lichess low-time alarm ([forum post](https://lichess.org/forum/redirect/post/1SL2qHTI)).
  Community analyses, not peer-reviewed. Abstract.

### Lit review 2026-11-12: personalization, ceilings, discriminators, benchmark tracks

Four parallel searches; papers were read through a fetch summarizer, so "full" means the full text was summarized,
not read line by line. Numbers marked (unverified) need a direct check before they go in a paper.

**The Allie test-set protocol, reproduced exactly.** Of the 18,239 games (1,239,353 positions), keep plies ≥ 10 and
cut the rest of the game at the first ply where the player to move has under 30 s before moving (clock rebuilt as
base − think + increment): 884,049 positions, the published count (checked twice, independently). Sources: Allie
(arXiv:2410.03893: "discard the first 5 moves … any moves made under time pressure"), Chessformer (arXiv:2605.19091:
"positions that occur after the first time a player has fewer than 30 seconds"), and the y0mingzhang/allie repo
(`src/allie/eval/maia3/aggregate.py`: `(ply >= 10) & (ply < pressure)`). Our earlier 56.65% top-1 for Maia-3-79M
(`lab/results/top1_allie_maia3-79m.json`) is on all 1.24M positions, not this filter. Published top-1: Maia-2 52.0,
Allie 55.7 / 55.9 (search), Maia-3 5M / 23M / 79M 55.4 / 56.6 / 57.1. ChessMimic (Johnson, arXiv:2606.04473) rebuilt
the set from public games (849,730 positions) and gets 57.3 for Maia-3-79M. Nobody reports NLL or perplexity, or a
personalized result, on this set. **The protocol drops plies 0–9, where 98% of our repeat visits (the repertoire
signal) fall.**

**Individual-player models.**
- McIlroy-Young et al., KDD 2022, [arXiv:2008.10086](https://arxiv.org/abs/2008.10086): per-player fine-tunes of
  Maia-1900; gain over Maia from −3.0 pt (1k games) to +5.2 pt (40k games). Opening perplexity 0.92 vs 2.40,
  non-opening 1.95 vs 2.15: most of the personal signal is the opening. Random 80/10/10 within-player split plus a
  "future month". Full.
- Behavioral stylometry, NeurIPS 2021, [arXiv:2208.01366](https://arxiv.org/abs/2208.01366): 98% identification
  among ~2,844 players from 100 games; 55% from moves at ply ≥ 30 only (unverified). A ready-made player-conditional
  discriminator, never used as a training signal. Skimmed.
- Maia4All (Tang et al. 2025, [arXiv:2507.21488](https://arxiv.org/abs/2507.21488)): Maia-2 player embeddings from
  prototypes; unseen players, first 10 plies dropped: 0.5146 → 0.5322 at ~20 games, 0.5381 at ~2,500. Flat after
  20 games. Full.
- Sogliuzzo, Rautureau, Piette 2026 ([arXiv:2605.11893](https://arxiv.org/abs/2605.11893)): Maia-2 embeddings for 16
  champions, +0.5 pt; argue top-1 is a poor style metric. Full.
- Carlson 2026, *Matilda* ([arXiv:2606.25176](https://arxiv.org/abs/2606.25176)): a zero-initialised residual over
  frozen Maia-3 plus Stockfish search features, 3000+ Elo games; top-1 60.1 → 66.1, of which player-style embeddings
  add +0.2 pt. Abstract (one subagent misreported it as a 32-d z on ~500 decisions per player; corrected here).
- **Gap:** no significant personalized gain over Maia-3 on held-out players; no phase breakdown on a strong base.

**Ceilings.** No paper measures a noise ceiling for human move prediction in chess or Go (no same-player repeat
agreement, no cross-player agreement at identical positions). Maia-2 and Chessformer call the noise "large" /
"aleatoric" without a number. Framing tool: Fudenberg, Kleinberg, Liang, Mullainathan, *Measuring the Completeness of
Economic Models*, JPE 2022 ([arXiv:1910.07022](https://arxiv.org/abs/1910.07022)): completeness = error reduction
achieved / achievable. Abstract. Saturation evidence is ablation-level only: Allie 355M → 774M +0.3 pt; Maia-3 23M →
79M +0.5 pt; ChessMimic finds the headroom grows with rating (0.3 pt below 1000, 2.2 pt above 2200).

**Discriminators and adversarial imitation.**
- Abramson et al. 2020, *Imitating Interactive Intelligence* ([arXiv:2012.05672](https://arxiv.org/abs/2012.05672)):
  GAIL alone never left its initialisation; BC + GAIL summed worked ("better at reshaping a behavioural prior than
  structuring it from scratch"), human-rated success 23.6% → 33.9%, **but BC validation log-prob got worse**.
  Adversarial pressure can trade our metric for realism. Full.
- InfoGAIL ([arXiv:1703.08840](https://arxiv.org/abs/1703.08840)) and VAE-GAIL (Wang et al. 2017,
  [arXiv:1707.02747](https://arxiv.org/abs/1707.02747)): plain GAIL mode-collapses; a latent from an encoder (VAE-GAIL
  is structurally our set encoder + D) or an MI term fixes it. Gains come from closed-loop compounding error. Full.
- Lu et al. 2022, *Imitation Is Not Enough* ([arXiv:2212.11419](https://arxiv.org/abs/2212.11419)): MGAIL helps a
  little; simple rewards plus a BC anchor help more. Full (table).
- AlphaStar: KL to the supervised policy plus pseudo-rewards on distance to a human statistic z (build order) — the
  precedent for matching style statistics without a discriminator. Secondary sources.
- Devlin et al. 2021, Navigation Turing Test ([arXiv:2105.09637](https://arxiv.org/abs/2105.09637)): learned human-vs-
  agent classifiers separate well but rank agents by human-likeness poorly. Abstract.
- Chess: Chidambaram & Qi 2017 style-transfer WGAN ([arXiv:1702.06762](https://arxiv.org/abs/1702.06762)), no numbers
  (abstract); Eisma et al. 2024, *Turing tests in chess* (doi:10.1016/j.chbr.2024.100496): Maia often judged human
  (abstract). piKL (Jacob et al. 2022, [arXiv:2112.07544](https://arxiv.org/abs/2112.07544)): search anchored to
  Maia, top-1 53.2 → ~54.3 at 1900 (approximate, parsed table).
- **Gap:** no GAIL/AIRL on a Maia-class policy; no player-conditional discriminator as a training signal; no
  statistic matching in chess.

**Group latents and the KL charge (v0g).** Neural Statistician (Edwards & Storkey, ICLR 2017,
[arXiv:1606.02185](https://arxiv.org/abs/1606.02185)) and ML-VAE (Bouchacourt et al., AAAI 2018,
[arXiv:1705.08841](https://arxiv.org/abs/1705.08841)) charge the group latent's KL once per group. The Variational
Homoencoder (Hewitt et al., UAI 2018, [arXiv:1807.08919](https://arxiv.org/abs/1807.08919)) trains per element with
the shared-latent KL divided by the size of the full set and a context that excludes the target: with a strong
PixelCNN decoder its context KL stays large where resampling alone collapses (numbers unverified; the abstract page
does not show them). This is the published form of v0g's `--kl-per-player`.

**Precedents for a personalized track.**
- Dynamic evaluation (Krause et al., ICML 2018, [arXiv:1709.07432](https://arxiv.org/abs/1709.07432)): "+ dynamic
  eval" rows beside the static model on fixed LM benchmarks, conditioning only on already-predicted elements. Full.
- LaMP (Salemi et al., [arXiv:2304.11406](https://arxiv.org/abs/2304.11406)): user-split and time-split tracks
  side by side, older items as the profile, random-profile and recency baselines. Full.
- Speech: per-speaker i-vector adaptation reported against a speaker-independent baseline on Switchboard (Saon et
  al., ASRU 2013); LibriSpeech's test sets are drawn by speaker so test-time adaptation is possible (Panayotov et
  al., ICASSP 2015, full).
- Recommendation leakage: Ji et al., TOIS 2023 ([arXiv:2010.11060](https://arxiv.org/abs/2010.11060)): context must
  precede the test item on the global timeline, or rankings change. Abstract.
- No precedent for adding recovered player identity or history to an existing chess test set.
- Safeguards to adopt: global-timeline causality; same model in both rows; matched-information baselines (random /
  wrong-player history, recency, repertoire table); coverage and history-size strata; validation-only tuning;
  release game IDs and fetch cutoffs.

### Registered: n = 1 coaching study (2026-11-15)

`docs/PROTOCOL_N1.md`: does simulation-based coaching speed up one player's improvement? The primary outcomes are
(1) the participant's blitz rating vs a synthetic control of look-alike Lichess players, with placebo inference, and
(2) the miss rate per opportunity of a coin-flip-assigned coached weakness vs an uncoached one. Baseline is ≥ 40
games per colour; coaching is gated on the simulation fidelity and forecast tests. Both gates reported 2026-11-16 (§8): simulation does not beat
shrunk real-game counts, so weaknesses are ranked from real games.

### Hypotheses: what proper scoring misses (logged 2026-11-12, untested)

Log-prob is a proper scoring rule: no model beats the true per-position distribution on top-1 or nats, so
adversarial training cannot raise the predictive noise ceiling. It can target a second ceiling, realism of sampled
play, where per-move log-prob is weak for two reasons. (1) Forward KL covers mass: it punishes missing a human move
hard and spare mass on moves humans never play lightly. §7's blunder tail is this gap: sampled Maia blunders 7.6% vs
humans 6.75%, and fixing it with temperature costs log-prob. (2) Evidence adds up: within-game structure worth
~0.01 nats per move is lost next to a ~1.3 nat per-move loss, but is ~0.4 nats per 40-move game, which a game-level
discriminator can detect. So part of the "aleatoric noise" seen by a history-limited model may be per-game or
per-player state: which plausible alternative a player picks is correlated across the game. Baseline hint: the
moves-only discriminator gets 0.53 for Maia vs human against 0.51 for human vs human (§8, Phase 0).

- **H1 (precision).** Among models within ~0.005 nats/move of each other, a game-level discriminator still separates
  them, ranked by blunder-tail excess. Cheap test, no training: self-play from Maia-3 at T = 1, T = 0.87 and rating
  input 1700, scored by `metrics.discriminator`.
- **H2 (hidden state).** D separates real games from Maia self-play better on whole games than on the same games
  with moves shuffled across games at matched positions or phase; the drop is within-game structure that per-move
  scoring underweights.
- **H3 (the adversarial rung).** A BC-anchored adversarial or statistic-matching fine-tune (Abramson-style BC + D;
  AlphaStar-style statistic rewards) lowers D accuracy and the blunder excess at a cost of ≤ 0.002 nats/move.
- **H4 (diversity, logged 2026-11-13).** A discriminator that sees a *set* of a player's games (the minibatch-
  discrimination / PacGAN idea) penalises a monolithic imitator even when each game is convincing on its own. A
  collapsed policy is too predictable across games: it plays the same move on every repeat visit, the same
  time-use regime every game, the same length of game. What D has to match is the *right* amount of predictability,
  which we have measured: humans replay their own earlier move on 77% of repeat visits on the Allie test set
  (mode-consistency), move choice carries no game-level state (§8 in-game state), while think time (9% game share)
  and error size (3.4%) vary strongly game to game. Predictions: argmax Maia is too consistent on repeats (100%),
  T = 1 sampling is too inconsistent, and both have too little game-to-game variance in time use. Cheap test inside
  (d): self-play several games per synthetic player from shared openings, then compare repeat-consistency and the
  game-share of think time and error size against the human values.

If these hold, the paper has three tracks: population predictive (top-1 / nats on the Allie test set), personalized
predictive (the same with prior history) and generative realism (generic and player-conditional D, blunder-tail
excess).
