# b5.406 notes

1. **Description.** Fires on squares attacked by one of the side-to-move's own pawns. It fires harder when two pawns cover the square, when the enemy also attacks the square or has a man standing on it (so a pawn capture is possible), and when our other pieces do not also guard it. It is stronger in higher-rated games.

2. **Evidence and what I tried.** "Our pawn attacks this square" is close to a necessary condition: only 0.002% of dev tokens without it fire. Among tokens with it, 40% fire with one attacking pawn and 78% with two. Inside that set the main graded terms are these:
   - enemy attackers on the square raise the fire rate from 0.24 to 0.77;
   - an enemy piece on the square raises it to about 0.85;
   - each extra guard of our own lowers it (0.48, 0.32, then 0.22);
   - our rook, queen or king on the square almost silences it;
   - rating band adds a clear position-level gain (fire rate 0.25 at 700, 0.50 at 2300).

   Smaller terms are relative rank 4 or higher, Maia move mass onto the square, the blockade and pawn-push squares, endgame (negative), and an attacking pawn that is blocked by our own man (negative). A linear fit and a depth-4 tree found nothing new beyond these terms.

3. **Final dev metrics** (9 scoring runs): Spearman 0.085, spearman_pos 0.727, AUROC 0.982, F1 0.700. The baseline concept pair scored F1 0.315.

4. **New concepts.**
   - `our_pawn_attacks(p, sq)` counts our pawns that attack sq.
   - `pawn_attackers_blocked_by_own(p, sq)` counts the attacking pawns that are stuck behind our own man.

5. **Caveats.**
   - The overall Spearman is low only because all tokens off our pawns' attack squares score a constant.
   - Inside the set of attacked squares, about 60% of the variance is unexplained. Misses include our own outposted piece or chain pawn that a pawn supports. False positives include empty squares that both sides contest heavily.
   - It is not polysemantic: it has one trigger, graded by context.
   - Confidence: high for the core trigger, medium for the grading.
