# b6.1739 notes

1. **Description:** fires on squares the opponent attacks (empty, or holding our king/rook), most strongly on the square
   in front of our advanced pawn (promotion square highest) and on our king's square when its flight squares are
   covered; the whole map is turned up in low-rated, lopsided, in-check, loose-material, long piece-rich games.

2. **Evidence and attempts.** Top examples are promotion squares in front of our 7th-rank pawn (h8, b8, h1, c1) and
   push squares of advanced pawns (d5-d4, e4-e5), plus many secondary firing squares that are simply enemy-controlled.
   Token-level: enemy attacker count is the strongest single feature (AUROC 0.72; empty & attacked 25% fire vs 3-4%
   unattacked); empty push squares attacked by them fire 42%, empty promotion squares 100%. Own king square fires
   with the number of attacked king-ring squares (AUROC 0.79), but only with material on the board. Position-level
   rate is dominated by rating band (Spearman -0.51), then |eval|, check, our hanging pieces. P/B/Q attackers count
   more than R/K; our pawn/knight guards damp it. Tried neighbourhood-pooled attack counts (worse than plain count),
   an endgame penalty (no gain), weaker band weight (worse). 11 scoring runs.

3. **Final dev metrics:** Spearman 0.198, spearman_pos 0.450, AUROC 0.860, F1 0.502 (baseline pair F1 0.364).

4. **New concepts:**
   - `king_ring_attacked(p, sq)`: number of squares around sq attacked by the opponent.
   - `our_hanging(p)`: number of our pieces attacked and undefended (cached per position).

5. **Caveats:** diffuse, map-like neuron; many enemy-attacked squares stay silent and the detector cannot say which
   (FPs concentrate in sparse endgames with checking queens). Position-level gain is only partly explained (band,
   eval, check). Looks like one theme ("enemy control / where our moves are unsafe") with pawn-push and king-safety
   specialisations rather than unrelated triggers: mildly polysemantic at most. Confidence: medium-low.
