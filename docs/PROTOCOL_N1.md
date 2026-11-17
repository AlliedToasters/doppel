# n = 1 protocol: does simulation-based coaching speed up one player's improvement?

Registered 2026-11-15, before any coaching exists and before any baseline analysis. Status: **baseline in progress**.
Changes after this date go in the deviation log at the end, dated, with the reason. Results are reported whatever
they show.

## Question

Does coaching from a player's Doppel (weaknesses ranked by simulated frequency × cost) make that player improve
faster than they would have anyway, and does it cause the specific changes it targets?

- **H1 (rating).** After coaching starts, the participant's Lichess blitz rating rises above the path predicted by a
  synthetic control built from look-alike players.
- **H2 (mechanism).** The targeted weakness's miss rate falls during its coaching phase, while an untreated weakness's
  rate does not.

H2 is the stronger test. Rating alone cannot separate coaching from practice, motivation or luck; a drop in the
coached weakness next to an untouched control weakness can.

## Participant

One player, the author. New to Lichess blitz (first rated blitz game 2026-11-13), with prior Lichess rapid
experience (rapid rating well above the blitz rating). No blinding is possible: the participant knows the phase.

## Games that count

Rated 5+3 blitz on Lichess against humans, using the corpus filter (`lab/doppel_lab/corpus.py` `keep`: rated blitz,
no BOT, Normal or Time-forfeit ending, clocks recorded). Positions are scored from ply 10. Games played inside the
Doppel app count only if they pass the same filter. Each game's phase is set by its start time.

## Phases

| Phase | Starts | Ends | Participant does |
|---|---|---|---|
| **Baseline** | 2026-11-13 | at ≥ 40 games per colour (≈ 80 games) | Plays normally; keeps study habits steady; sees **no** Doppel analysis of their own games |
| **Phase A** | when the coaching system and the gates below are ready | after 100 games | Coaching on weakness A only |
| **Phase B** | end of phase A | after 100 games | Coaching on A and B |

**Gates before phase A.** Coaching starts only after the simulation fidelity check (`lab/doppel_lab/sim_fidelity.py`)
and the forecast test (weakness rates in games 41–100 predicted from simulation vs from the first 40 games alone) are
reported. If simulation does not beat real-game counts, weaknesses are ranked from real-game evidence only and this
is logged as a deviation. Time between the baseline and phase A is logged; games played in that gap count as
extended baseline.

## Primary outcome 1: rating vs a synthetic control (H1)

- **Measure.** The participant's Lichess blitz rating after each game.
- **Donor pool.** Players from the Lichess monthly dumps who, like the participant, (a) started rated blitz with
  prior Lichess games in another time control, (b) played at least as many blitz games as the participant over the
  study, at a similar pace (games per week), and (c) started within 12 months of the participant.
- **Synthetic control.** Non-negative donor weights summing to 1, fit to the participant's baseline path (rating
  after each game, by game index) plus pace and other-time-control rating. **The weights are frozen at the end of the
  baseline**, before any coaching.
- **Effect.** The participant's rating minus the synthetic control's, at the end of phase A and of phase B.
- **Inference.** Placebo tests: refit the same method with each of ≥ 500 donors as the "treated" player and record
  their post-baseline gaps. **H1 is supported if the participant's gap exceeds the 95th percentile of placebo
  gaps**, among placebos with a pre-period fit no worse than 2× the participant's.
- **Expected precision.** One player's rating is known to about ±350/√n Elo after n games (≈ ±35 at 100). An
  effect under ~50 Elo per phase is unlikely to show here; H2 carries the mechanism.

## Primary outcome 2: targeted weakness rates (H2)

- **Weakness taxonomy (fixed now).**
  - **Blunder:** a move by the participant from ply 10 on that loses ≥ 0.2 expected score (Stockfish, 20k nodes,
    the same definition as the fidelity check).
  - **Blunder type:** tagged by the lichess-puzzler motif tagger applied to the refutation.
  - **Time phase:** the clock, split into > 60 s, 30–60 s and < 30 s remaining.
- **Rate decomposition.** For each weakness, rate = how often the opportunity arises × the probability of missing it
  when it does. H2 is tested on the **miss probability per opportunity**, so a change in openings or opponents can't
  fake an effect.
- **Choosing A and B.** At the end of the baseline, rank weaknesses by estimated cost (expected points lost per 100
  games). Take the top two that each have ≥ 30 baseline opportunities. **A coin flip assigns which is A**; the other
  is B. The flip is recorded with its date.
- **Test.** Difference in differences: (A's miss rate in phase A − baseline) − (B's miss rate in phase A − baseline),
  with a 95% CI from a bootstrap over games. **H2 is supported if the CI is below zero.** Phase B is the replication:
  B's rate should then fall too.

## Logged alongside

- Games per day and time of day.
- Study time outside Lichess games, as a one-line daily note.
- Streaks: whether the previous game was lost.
- Rating of opponents.
- Any coaching material seen, with date.

## Analysis hygiene

- No look at baseline weakness rates until the baseline ends; the A/B choice uses only the rule above.
- The analysis scripts are written and run on donor data before the participant's post-baseline data is analysed.
- Every number in a write-up comes from a script in `lab/` with its output in `lab/results/`.

## Deviation log

| Date | Change | Reason |
|---|---|---|
| 2026-11-16 | Gates reported. Weaknesses will be ranked by baseline counts shrunk toward the crowd at the participant's rating (empirical Bayes), not by simulation. | Fidelity check: a player's first 40 real games predict their later blunder rate better than 40 simulated games (r 0.56 vs 0.25). Forecast test on 200 held-out players: shrinking toward a simulation-based prior ties shrinking toward the mean (36–37% of explainable variance either way). Tested on overall blunder rate (≥ 0.2 and ≥ 0.3), not yet per motif or clock phase. No participant data was used. SCIENCE.md §8. |
