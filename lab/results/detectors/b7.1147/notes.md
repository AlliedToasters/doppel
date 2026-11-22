# b7.1147 notes

1. **Description:** fires on a contested square that their queen (or a rook) attacks and we cover at least twice. It is
the place to challenge or trade off their heavy piece (often the square where the two queens meet), and it is stronger
when we can move a like piece there, Maia wants to play there, or we are ahead in material.

2. **Evidence / what I tried:** Top examples are almost all squares both queens attack (e4 between Qe1/Qf3, b5 between
Qa5/Qc4, ...), plus rook-vs-rook squares (a8 with Ra7/Re8). On dev, squares attacked by both queens fire 22-24% of the time
against a 1.3% base rate. Their queen attacking the square is the biggest single term. A logistic probe over attacker types,
occupancy, move-to features and context put the weight on these: their queen attacks; we attack it 2+ times; Maia's
move-target mass; their queen sits on the square; our own piece there (negative); their pawn or king guards it (negative);
a like-for-like move is available. A 26-feature GBM ceiling was about F1 0.53, and a gated logistic reached 0.54 held out.
The hand-written rule reaches 0.51. Runs, as F1: heavy-contact product 0.296, typed attacker terms 0.454, exchange-offer
term 0.450 (no gain), pawn-guard penalty and material 0.480, compact rule 0.512, heavy-twin standoff 0.513. A GELU-wrapped
output hurt AUROC and spearman_pos, so the raw score is kept.

3. **Final dev:** Spearman -0.285 (all tokens; negatives sit at a constant floor), spearman_pos 0.262, AUROC 0.972,
F1 0.513 (thr 3.15). Baseline pair F1 0.275. 8 scoring runs.

4. **New concepts:** `like_for_like_move(p, sq)`: we have a legal move to sq with a piece type (N/B/R/Q) that already
attacks sq from their side, so playing it offers a trade of like pieces.

5. **Caveats:** Precision is the limit (about 2.5k false positives at the best threshold). Many squares their queen attacks
twice over look identical in these terms but stay silent. The neuron probably also reads tactical or exchange outcomes that
the attack sets do not capture. Firing magnitude is poorly ranked (spearman_pos 0.26). It is not clearly polysemantic: queen
vs queen, rook vs rook and other like-piece contact all fit one theme, "where our and their pieces trade", with their queen
as the strongest trigger. Confidence: medium.
