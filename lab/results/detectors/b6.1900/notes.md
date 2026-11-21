# b6.1900

1. **Description.** The neuron fires mainly on empty back-rank squares in our rook's lane: the rooks are not connected yet (a Q, B, R or K stands between them), the rook is often still in its corner, and the square has a piece in front of it. It also fires on empty squares between the enemy king and rook, plus a weaker, diffuse signal on contested squares.

2. **Evidence and what I tried.** It fires on 3.7% of dev tokens. About 39% of the firings sit on our first rank, where the fire rate is 11%, and on those squares it fires almost only when they are empty (18%). I grouped empty squares by their nearest neighbour on each side along the rank. Our rook with our Q, B or R on the far side gives a 39-44% fire rate, and rook to our king gives 21%. Between the enemy rook and enemy king (mostly f8/g8 or b8-d8 next to an uncastled king) gives 15%. Everything else is 1-10%. Inside the lane, firing goes up when the rook is in its corner, sits close to the square, a piece stands directly in front of the square, and the enemy attacks it. Firing goes down when the square is in Stockfish's first two PV plies.

   The rest of the firing (about 60%) is spread over all ranks. It tracks how contested the square is (attacked by both sides, stronger when occupied, and pawns), squares next to our king, and later plies. Weights came from a logistic fit on a dev subsample, rounded and kept as readable terms.

   Scoring path: rank-1 empties gave AUROC 0.651 and F1 0.286. The rook lane alone gave 0.656 and 0.291. Adding the contested terms gave 0.799 and 0.306. The joint terms gave 0.866 and 0.337. Refining the lane and reweighting gave 0.870 and 0.353. A GELU output mapping lowered AUROC to 0.64, so I reverted it.

3. **Final dev metrics.** Spearman -0.082, spearman_pos 0.268, AUROC 0.870, F1 0.353 (baseline concept pair 0.239). 10 scoring runs.

4. **New concepts.**
   - `rank_neighbours(board, sq)`: the nearest occupied square to the left and to the right along the rank.
   - `rook_lane(p, sq)`: whether an empty square is in our rook's lane along its rank, graded by the friendly piece that closes the lane off, with that rook's square. Also flags a square between the enemy rook and the enemy king.

5. **Caveats.**
   - Precision inside the lane is only about 33%, and I found no clean position-level gate for it. Firing peaks around ply 20-40 and with more pieces on the board, and drops in the opening when only b1/g1 have been vacated.
   - The off-back-rank firings are diffuse and only roughly captured by the contested-square terms.
   - Overall Spearman is negative because the negatives are ordered poorly.
   - It looks moderately polysemantic: back-rank rook connection, plus a general "contested square" signal.
   - Confidence: medium on the main trigger, low on the rest.
