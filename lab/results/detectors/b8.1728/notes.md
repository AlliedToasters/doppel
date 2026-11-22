# b8.1728 notes

## 1. Description
An always-on "weak-player level + our sphere of action" neuron: its overall level falls with rating (band alone
gives dev Spearman 0.378; about -0.25 per 400 Elo, steeper at 2300), and within a position it is highest on the side to
move's own men (minors/queen/pawns, free rooks; in the opening especially undeveloped minors and unpushed pawns), on
squares our legal moves or pawn pushes reach and on Maia's expected move squares, and lowest on squares the opponent
guards and on enemy rooks/queen/king.

## 2. Evidence and what I tried
Fires (>0.1) on 100% of tokens, so only Spearman matters. Dev variance split: ~30% between positions, ~70% within.
Band explains ~46% of the position-mean variance (pos-mean vs band rho -0.68). Band-only detector: 0.3775.
Token regression showed att_us (rho 0.47) beats band on its own; band + att_us + att_them -> 0.61. Residuals pointed to
legal-move destinations (reachable, rather than raw guard count, saturating at 1-2 moves), pawn-push squares,
Maia from/to mass, own-piece type (rook only if mobile), and an opening effect (home-square minors/pawns up, developed
minors down). Within-position slope of activation on the square part falls from 1.12 (700) to 0.83 (2300), so the
square part is scaled by (1 - 0.075 band_z). Small context terms: eval (worse for us -> higher), n_pieces,
Maia entropy, castling rights. Runs: band 0.378, v1 0.683, v2 (legal targets, Maia from) 0.741, v3 0.743,
v4 (reach bool, rook mobility, king) 0.7585, v5 (band-scaled squares) 0.7596, v6 (home squares) 0.7612;
recalibrating band table / adding check penalty made it slightly worse (reverted).

## 3. Final dev metrics
spearman_pos 0.7612 (band only 0.3775); AUROC NaN, F1 1.0 (meaningless: fire rate 100%). 10 scoring runs.

## 4. New concepts
- on_home_square(board, sq): the man on sq stands on its own starting square from the initial position.
- opening_weight(ply, span=40): 1 at game start, falling linearly to 0 by ply span.

## 5. Caveats
About 40% of variance is still unexplained, including ~1/3 of the between-position level (not rating, eval or
phase; possibly game-history/style cues Maia reads that we cannot see from the board). The weights are a linear
fit of ~20 terms, so it reads as a graded "our territory/activity" map, not a crisp rule. Not polysemantic in the
usual sense (one smooth map), but it mixes a global rating code with a square code. Confidence: medium.
