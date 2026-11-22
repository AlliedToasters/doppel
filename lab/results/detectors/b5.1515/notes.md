# b5.1515 notes

(1) Fires on the opponent's pieces (never pawns, never our own men, never empty squares): any developed, mobile
enemy knight/bishop/rook/queen in contact with our men, strongest for knights and when the opponent has few pieces
left; undeveloped pieces on their home squares stay silent in the opening and light up as the game goes on.

(2) Evidence: 100% of firings are on enemy non-pawn pieces (fire rate: N 68%, B 51%, Q 52%, R 37%, K 15%; pawns,
own pieces, empty squares 0%). Within enemy pieces the main split is home square / back rank (20% vs 75%+ elsewhere),
modulated by ply (home pieces 0% in the first 15 plies, ~60-90% after ply 60), mobility (AUROC 0.67, kings 0.82),
contact (attacks one of our men or is attacked: 31% -> 75%+), and how many officers the opponent has left (fewer ->
each fires more; a mild per-position competition). Tried: eval magnitude, Maia capture mass on the piece / on
another enemy piece, check-giving, uncastled opponent for all pieces (none helped beyond noise); uncastled opponent
for home pieces gave a tiny F1 gain. A gradient-boosted model on the same features reaches ~0.91 AUROC within enemy
pieces vs ~0.887 for this linear rule, so little is left in these features.

(3) Final dev: AUROC 0.9931, F1 0.8021 (baseline pair 0.34), spearman_pos 0.734 (all-token Spearman 0.09 is
meaningless here since non-piece tokens are a constant). 8 scoring runs (the last re-confirmed the final file after removing an unused helper).

(4) New concepts:
- on_home_square(p, sq): the piece stands on one of its side's original starting squares (undeveloped).
- n_officers(p, ours): knights+bishops+rooks+queens one side still has.

(5) Caveats: magnitude is only moderately explained (spearman_pos 0.73); misses home rooks on a8/h8 in the
middlegame that fire anyway, and false positives on active enemy pieces in positions where another enemy piece
takes the activation. Not polysemantic: one trigger (enemy piece presence/activity) with graded modifiers.
Confidence: high for "enemy non-pawn piece, developed", medium for the modifiers.
