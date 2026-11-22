# b8.1140

**(1) Description.** A dense, always-on "mover's live squares" neuron. Its level across the whole board is set mostly by
rating (higher for weaker players: band alone gives dev Spearman 0.382) and is raised in the first ~12 plies.
On top of that level, each square scores higher when it is in play for the side to move: our minor pieces and pawns,
squares we attack or can legally move to, enemy minors and pawns, and above all the from/to squares of Maia's
expected moves. It scores lower on our king and its neighbours, our back rank, enemy rooks and queens, squares the
opponent piles onto, and the far enemy ranks.

**(2) Evidence and what I tried.** The activation is > 0.1 on 100% of tokens, so I scored only by Spearman. About 30%
of the dev variance is between positions and 70% within a position. Between positions: the position mean falls with
band (3.51 at 700 down to 2.49 at 2300). After band, ply matters most (+0.32 at plies 10-16, about 0 after ply 24).
Smaller effects: an obvious top move, a better eval and many legal moves all lower the level. Within a position, the
raw correlations were att_us +0.51, legal destination +0.42, Maia move distance -0.39, relative rank -0.32 and
att_them -0.27. The att_us and legal-destination effects saturate at one, so I used indicators for them. Residuals
then showed effects by piece type: our king and rooks are low, enemy minors and pawns are high, enemy rooks and queen
are low. I set the weights by eye and checked them with a 12-term least-squares fit on a 1500-position dev sample.
- Band only: 0.382.
- Square terms only, without the position level: 0.591.
- First combined version: 0.670.
- Adding piece types, a sqrt of Maia's move mass, the back-rank and king-neighbour terms, the opening ramp, and the
  eval and mobility terms brought it to 0.734.

**(3) Final dev metrics.** Spearman / spearman_pos 0.734. AUROC NaN and F1 1.0, both meaningless because the neuron fires
on 100% of tokens. 10 scoring runs; the last one was the square-only ablation.

**(4) New concepts.** None. Everything is built from Pos methods. The only extra quantity is the per-position legal
move count, which is `sum(len(v) for v in p.legal_to().values())`.

**(5) Caveats.** About 0.11 of the residual variance is still between positions, and nothing I tried explains it:
eval, check, number of legal moves and ring pressure each give only |r| around 0.2. The square term is a blend of
"controlled or reachable by us", "on Maia's candidate moves" and "piece-type/rank priors". It reads as one broad
"where the mover's play is" field, not several unrelated triggers, so it does not look polysemantic in the usual
sense. It is diffuse, though, and it is plausibly a rating-conditioned bias channel whose square pattern partly
copies Maia's policy. The detector reads Maia's top-5 moves, so part of this explanation is circular. Confidence:
medium for the rating/opening level, low-medium for the chess meaning of the square pattern.
