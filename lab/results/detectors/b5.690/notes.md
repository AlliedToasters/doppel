# b5.690 notes

1. **Description.** Enemy-held territory: fires on squares (empty or holding an enemy man) that the opponent controls and the side to move does not contest, strongest on enemy-occupied squares deep in the opponent's half; never on our own pieces.

2. **Evidence and what I tried.** Fire rate 27.7%. Firing rate rises with enemy attackers (5% at 0 attackers, 36% at 1, 67% at 2, 83% at 3) and collapses as soon as we attack the square (50% with no attacker of ours, 8% with one, 1% with two); a pawn of ours guarding it is the strongest veto. Enemy-occupied squares we do not attack fire 63-96%; our own pieces fire 0.6%. Relative rank 5-7 fires ~60%, ranks 0-2 ~2%. Position level: more men on the board means more firing (r 0.75 for the fraction of firing squares). v1 (enemy attackers minus ours + enemy piece + rank) already got AUROC 0.964 / F1 0.844. Tried: neighbour-area control (small gain), our pawn count and ply (worse), enemy slider x-ray reach through one blocker (worse), capping control and their-king distance (neutral), GELU output (Spearman 0.57 but AUROC 0.928, rejected). Best gain came from weighting attackers by piece type (ours: pawn 3.4, others ~2; theirs: pawn 2.0, rook 1.9, king 1.8, queen 1.5, bishop 1.3, knight 1.0). A gradient-boosted model on the same features reaches ~0.969 AUROC, so the rule is close to that ceiling.

3. **Final dev metrics.** Spearman 0.3835, spearman_pos 0.7609, AUROC 0.9666, F1 0.8559 (baseline concept pair F1 0.571). 11 scoring runs.

4. **New concepts.**
   - `neighbour_control(p, sq, by_us)`: fraction of the king-step neighbours of sq that a side attacks.
   - `weighted_control(p, sq, by_us, w)`: sum of per-piece-type weights over a side's attackers of sq.

5. **Caveats.** Misses: some lone-rook files in bare endgames, and our king when in check (both fire). Position-level magnitude (how hard the whole enemy camp lights up) is only partly explained; queenless and late positions fire weaker than the rule predicts. Overall Spearman is low because non-firing tokens are ranked arbitrarily. Looks monosemantic (one graded territory/control concept). Confidence: high for the main rule, medium for the magnitude.
