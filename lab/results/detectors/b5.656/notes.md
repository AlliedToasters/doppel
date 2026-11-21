# b5.656 notes

1. **Description.** A diffuse "quiet, safe square" map: it fires on squares the enemy does not attack (especially not
   with a rook or queen) that are two or more squares from our own king, preferably on our 5th-7th rank and along
   our rooks' lines. Our own bishop on the square raises it, our own knight or an enemy pawn lowers it. The whole map
   runs hotter in 700-rated games and when we are in check.

2. **Evidence and what I tried.** The top examples look scattered: rook-rank squares, an undeveloped c1 bishop, a
   cluster on the queenside. The strongest single signal is "the enemy has no attacker on it" (fire rate 29% vs 4%).
   Next come distance from our king (almost never fires within 1 square of it; peaks 4-6 squares away), our rook's
   line (clear or x-ray), relative rank 5-7, and the occupant of the square. Band 700 roughly doubles the
   position-level fire rate (r = 0.50 with per-position fire rate). Being in check, the opening (ply < 10) and very
   full or very thin boards shift the whole map. Two-ply enemy reach, Maia move mass and SF PV gave nothing. I read
   the readable weights off a small logistic fit on about 20 concept features and rounded them. A GBM on the same raw
   features reaches about 0.888 AUROC, so some nonlinear structure is still unexplained.

3. **Final dev metrics** (run 10): Spearman 0.162, spearman_pos 0.451, AUROC 0.861, F1 0.474
   (baseline pair `empty & att_us>=1` F1 0.243).

4. **New concepts.** None beyond the library (an occupant-value table is detector-specific).

5. **Caveats.** It is a weighted sum of about 12 graded terms, not a crisp rule. Precision is only about 45% at the
   best threshold. It misses firings on squares attacked only by the enemy king or bishop, and firings in positions
   where an enemy piece has invaded next to our king (enemy Q on h2, B on f2). Polysemantic? It looks like one broad
   "safe, away from our king" field modulated by rating band rather than several unrelated triggers, but it is
   diffuse. Confidence: low-medium.
