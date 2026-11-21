# b7.946 notes

1. **Description.** A diffuse "the move is settled, this square is not part of it" neuron: it fires on many squares at once when Maia-3 is confident (one obvious move, few legal moves, a capture, late game, side to move clearly worse), and within such a position it picks squares far from the candidate moves, away from our own king, not hit by enemy pieces, and on our back rank, while our own pawns and king stay silent.

2. **Evidence and what I tried.** The variance is mostly at the position level. The oracle per-position mean alone gives AUROC 0.83. Position fire rate correlates with Maia top-1 probability (r +0.39), with top-5 entropy (-0.37), with legal-move count (-0.34, strongly non-linear below ~15 moves), with ply (+0.34) and with being in check (+0.30). Stockfish eval is asymmetric: lost positions fire twice as often as equal ones, and winning mating attacks are silent. Top-5 mass (how little probability is left outside the top 5) adds to entropy. Square level: the probability-weighted distance to Maia's top-5 move squares is the best term (within-position rho +0.34). Our pawns (7% fire rate) and our king (7%) are silent; our rooks, queens and knights fire. Enemy pieces are slightly suppressed and their king is boosted. When our king has left the back two ranks, our back rank lights up (59% vs 23%). I fitted weights with small logistic and least-squares fits on dev, then rounded them into readable terms. A gradient-boosting probe on the same features reaches only ~0.82-0.83 AUROC, so the remaining gap is features I did not find.

3. **Final dev metrics** (run 7): Spearman 0.3845, spearman_pos 0.478, AUROC 0.8155, F1 0.530 (baseline concept pair F1 0.217; always-on 0.32).

4. **New concepts.**
   - `maia_entropy(p)`: entropy of Maia's top-5 distribution.
   - `maia_move_distance(p, sq)`: probability-weighted Chebyshev distance from sq to the top-5 moves' from/to squares.
   - `material_balance(b, us)`: material of the side to move minus the opponent's, in pawns.

5. **Caveats.** About 20 weights, so this is more a graded score than a crisp rule. Position-level prediction caps out near Spearman 0.67 against the per-position fire rate, so the neuron probably reads an internal "confidence/settledness" signal we only proxy. The square-level pattern is weak (within-position rho ~0.28 vs ~0.39 for a GB probe; the missing part involves rank × piece × king-distance interactions). It does not look polysemantic: one diffuse context signal. Confidence: medium on the position-level story, low-medium on the square choice.
