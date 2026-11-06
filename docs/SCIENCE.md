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

Move choice itself is a research target, but at this time is only planned future work. Currently we use
off-the-shelf models like Maia, and plan to leverage other chess-playing models such as ChessMimic when we begin
modeling move choice. Until then, v0 distils move choice from Maia-3 (`doppel_lab/teacher.py`) and we check only
that it does not regress (top-1 and perplexity, `doppel_lab/top1.py`).

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

## 3. Features, ordered by compute cost

We pick features in order of cost. Cheap features are used everywhere; expensive ones only where the payoff is
measured.

| Tier | Feature | Cost (lab server) | Notes |
|---|---|---|---|
| 0 | Clocks, increment, ply, both ratings, legal-move count, in check | free | Already in the game record or one board call |
| 1 | **Material** balance (P 1, N 3, B 3, R 5, Q 9) | microseconds per position | Best signal per unit of compute |
| 2 | Maia-3 outputs: policy entropy, top-move probability, win/draw/loss | ~12,000 positions/s on the GPU (79M model) | Needs the warden; already stored for v0 (§3.2) |
| 3 | **Stockfish eval** at 50,000 nodes | ~49 ms per position per core (~depth 16) | Strongest single feature; see §3.1 for when it is worth it |
| 4 | Stockfish eval at 1,000,000 nodes | ~1 s per position per core (~depth 22) | Reserved for spot checks |

Costs measured 2026-11-06 with Stockfish 16 (about 2 M nodes/s per core) and the Maia-3 teacher job.

### 3.1 Stockfish eval

- **What it is:** an evaluation function. Stockfish searches ahead (alpha-beta with pruning, a transposition
  table, and an NNUE neural net as the static eval at the leaves) and reports a score in **centipawns** (cp),
  hundredths of a pawn, from White's view. Forced mates are stored as ±10,000.
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

- **Channels** (raw, per ply): clocks, think time and think fraction, legal moves, check (tier 0); material
  (tier 1); Maia-3 win/draw/loss, expected score, sharpness, policy entropy, top-move probability, played move's
  log-prob and rank (tier 2); Stockfish cp, winning chances, CPL, Lichess move class (tier 3).
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

**Finding, sharpness.** jk_182's formula (§8) depends on the draw probability: `(2 / log((1−W)(1−L)/(WL)))²` grows
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

## 7. Adding a capability

1. Write its section here first: target definition, metrics, and which priors, heuristics and machines it is
   scored against.
2. Write the label code in `lab/doppel_lab/` as a sidecar to the existing artifacts, joined on `game_id`, with
   tests in `lab/tests/`.
3. Record the prior and heuristic baselines in `lab/results/` before any neural work.
4. Add machine baselines or proxies.
5. Only then add a head to the Doppel model.

## 8. Literature

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
