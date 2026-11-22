# b8.738 notes

(1) **Description.** A broad, always-on "weak player / own-activity" level. The rating band alone gives dev
spearman_pos 0.388 (weaker players fire more). On top of rating, it is louder when the side to move is worse or lost,
when Maia's top 5 cover little probability, and on a calm board. Across the board it peaks on the from- and
to-squares of Maia's expected moves, on squares we control and can move to (pawn-advance squares, enemy men we
attack, our active N/B/R/Q), and in the enemy half. It is lower on enemy-guarded squares, our back rank and
king-step squares.

(2) **Evidence and what I tried.** It fires on 83% of tokens, so AUROC and F1 for "fires" carry no information and
spearman_pos is the metric. About 36% of the variance is between positions:
- Position means correlate with band at r = -0.62.
- Rating only (`-band_z`) gives spearman_pos 0.388 (detector_band.py).
- Within a band, the position level also follows the eval: about +0.8 when the side to move is at -10 or worse,
  ramping from about -4. It rises when Maia's top-5 mass is low or its entropy high, and falls with hanging pieces
  or captures available.

Within a position (64% of the variance), the strongest correlates are Maia move mass to and from the square
(r 0.37 / 0.31), our guard count (+0.39), enemy guard count (-0.29) and the number of legal moves to the square
(+0.32). The top examples are knight-fork targets (d5c7 and similar). These are the squares of the expected move,
not forks as such. Piece-type bonuses, the pawn-push and king-step terms, and a relative-rank profile added about
0.04. Using the square root of Maia mass added 0.012. Scaling the position-level terms down (they are over-weighted
under rank correlation) added about 0.04.

Decomposition with the final detector:
- Position-mean spearman: about 0.79.
- Median within-position spearman: about 0.69.
- Using the true level with my within-position terms gives about 0.81.

Trajectory: band 0.388, then linear Maia/guards 0.615, then position terms 0.650/0.678, then square terms 0.691,
sqrt mass 0.703, then rescaling 0.743.

(3) **Final dev:** spearman 0.7438, spearman_pos 0.7426, AUROC 0.8662, F1 0.907 (the always-on value). 14 scoring
runs.

(4) **New concepts.**
- `legal_from_count(p, sq)`: the number of our legal moves starting on sq.

(5) **Caveats.**
- The neuron is dense and graded. This is a sum of about 20 hand-set weights, not one crisp rule, and the weights
  were tuned by hand against dev scores.
- About half of the position-level variance after rating is unexplained. It is plausibly player or game style,
  which is consistent with the neuron separating personalised adapters.
- It depends on Maia's own outputs (maia_p5 / top-5 move squares), so part of the explanation is "where the model's
  move is", which is circular.
- Not obviously polysemantic. It looks like one diffuse "human-ish activity / uncertainty" signal scaled by rating.
- Confidence: medium.
