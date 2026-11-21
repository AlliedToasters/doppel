# b7.1015 notes

**(1) Description.** Fires on squares in the side-to-move's own camp (its first three ranks, mostly the 2nd and 3rd)
that belong to its short-range men: the king's square, its pawns and knights, empty squares one of its pawns can step
onto, and squares its knights cover. It stays quiet on squares held by its rooks or queen, squares next to its king,
and squares the enemy attacks. It is far stronger in low-rated games and in quiet positions.

**(2) Evidence and what I tried.** Fire rate by relative rank is 7.6% / 13.6% / 17.7% on ranks 1-3, then under 3%
from rank 4 up. Our king's square fires 36% (any rank) and never-attacked king squares more. Empty pawn-push targets
fire 30-52%. Empty squares covered by one of our knights fire 32% (49% with two), against 9% without. Squares covered
by our rooks or queen fire less (16% -> 9% -> 3% for 0/1/2 R/Q). So do squares next to our own king, and squares the
enemy hits (9% -> 3% -> 0.5%). Enemy pieces in our camp almost never fire. Rating band is the largest position-level
factor: on safe camp squares the fire rate goes 22% at 700, 10% at 1900 and 5% at 2300. Being in check, Maia having
one obvious move (high top-1 p) and many pieces on the board also suppress it. Path: v1 (camp & unattacked &
K/P/N/empty x band) gave F1 0.34. Adding pawn-push targets brought it to 0.39. A depth-5 tree showed that "push" and
"band" were the splits that mattered. A table of which piece types can reach the empty squares turned up the knight
term. A logistic fit on camp tokens, rounded into hand-readable log-odds terms, gave 0.44. Grading the outside-camp
squares instead of hard-zeroing them, and letting the king's square fire on any rank, lifted AUROC. The "any legal
move lands here" and piece-count terms then gave the final 0.449. Gating on enemy intruders or on Stockfish eval
added nothing.

**(3) Final dev metrics.** Spearman -0.050, AUROC 0.903, F1 0.449 (baseline `legal_to & rank3` 0.243; always-on
0.106).

**(4) Caveats.** Spearman is near zero by construction. The activation is a GELU output: enemy-half squares sit at
about -0.08 (deeply off) and own-camp non-firing squares at about -0.14 (near threshold). A monotone detector
therefore cannot rank the negatives correctly. Squashing the score through a GELU raises Spearman to about 0.23 but
costs AUROC (to 0.6-0.8), so I kept the monotone score. Within an eligible square the detector is right well under
half the time. Many firings depend on the position as a whole in a way I could not capture: whole camps light up in
some low-rated quiet positions and not in similar-looking ones. Some firings are missed: a lone king far up the
board, squares behind a bishop, and isolated empty squares at 1900+. The rule needs about 17 weights, rounded from a
logistic fit, which is more than the brief's guideline of about 10. The concept looks like one family ("our camp
squares owned by king, pawns and knights, in calm low-Elo positions"), not several unrelated triggers. The strong
Elo dependence suggests it partly encodes "a low-rated player's typical home-side moves". Confidence: medium on the
spatial and piece pattern, low on the exact gating.
