# b5.1178 notes

1. **Description.** Fires on the enemy's diagonal lines of pressure, i.e. squares an enemy bishop (or queen,
   diagonally) looks along in the middle of the board, most strongly when the diagonal runs into our king, queen,
   rook or bishop either directly or through one of our pawns (pin / x-ray), and also on our own advanced pieces
   (king, queen, rook) and on our centre pawns, more so in low-rated games.

2. **Evidence and what I tried.** Top examples are squares on an enemy bishop diagonal: c6/d5/e4 on Bb7-f3 (bishops
   facing each other), e7/d6 on Bf8-a3, c6 on Ba4-e8 (the pinned pawn in front of the king), f5 on Bh3-c8. The
   python-chess baseline (`att_them>=2 & rank6`, F1 0.14) was weak. Data analysis on dev: enemy bishop line_hit==2 gives
   a 13.5% fire rate against 2.8% base; our pawn pinned on an enemy bishop diagonal to our K/R/B fires 70-96%. About
   60% of firings are on our own pieces, mostly advanced ones (rank >= 4 in our frame: 20-35%, advanced king 74%).
   Empty diagonal squares on our first three ranks rarely fire unless the line hits a valuable piece; squares next
   to the enemy king that their bishop covers fire often. Own centre pawns fire mostly at the 700 band. I explored
   with decision trees and logistic regressions (LR ceiling on my features was about 0.40 F1) and then wrote a
   hand-weighted score: enemy_diagonal_pressure (walk each enemy B/Q diagonal, with a "loaded" bonus when it ends on
   our K/Q/R/B, directly or through one blocker), gated by rank for empty squares, plus terms for own advanced pieces
   by type, our king (especially in check), centre pawns, Maia move-to mass, back-rank / edge / distance to the enemy
   king penalties, endgame and rating. I tuned the weights offline on a dev subset. Scoring went 0.318, 0.331,
   0.351, 0.364, 0.374, 0.386, 0.393.

3. **Final dev metrics.** Spearman -0.390 (GELU dip; the raw score is not GELU-mapped), spearman_pos 0.285, AUROC
   0.920, F1 0.393 (10 scoring runs).

4. **New concepts.**
   - `enemy_diagonal_pressure(p)`: {square: score} along the enemy's bishop/queen diagonals up to the first man,
     with a bonus when the line ends on our K/Q/R/B directly or behind one blocker (pin / x-ray / discovered attack).

5. **Caveats.** The neuron is broad (fires in 74% of positions) and looks polysemantic: diagonal-pressure lines,
   our advanced heavy pieces / king, and our centre pawns at low rating are separate triggers. Many empty squares on
   enemy diagonals do not fire (precision about 0.35-0.4). Direct attacks on our queen fire less than the geometry
   suggests. The overall Spearman is negative because near-miss tokens sit at GELU's dip. Confidence: medium-low.
