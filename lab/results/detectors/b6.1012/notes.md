# b6.1012 notes

1. **Description.** Loose or stranded enemy men: an enemy piece that is undefended, stuck with little room on its own back rank (corner rooks, home king/knights) or deep in our half / touching our king. It fires much more in low-rated (700-1100) games and when the side to move is losing or short of material. In those games it also spreads onto empty enemy-half squares and our own unmoved back-rank rooks.

2. **Evidence and what I tried.** The fire rate on dev falls steeply with band: 10.3% of tokens at 700, 4.0% at 1100, 1.4% at 2300. Enemy pieces fire at 13-32% by type, against 1.7% for empty squares and about 2% for our pieces.
   - Within enemy pieces: undefended gives 29% vs 9%, own back rank 61% (at band 700), and next to our king 54-82%. Low mobility matters too: a band-700 rook with 0 free squares fires 90%.
   - A king or rook that keeps its castling right fires less at higher bands.
   - At band 700, empty-square firings cluster in lost positions (sf_cp of about -1000 or worse, or we have little material), and in the enemy half.
   - Built as an additive graded score. Weights came from a small logistic fit and were then rounded. A GBM over the same features plus the detector reaches F1 about 0.52. Its gain is spread over many small interactions (piece type x rank, mobility, ply, piece count); no single extra term was worth more than about 0.01.

3. **Final dev metrics** (run 11): Spearman -0.106, spearman_pos 0.309, AUROC 0.903, F1 0.437 (baseline pair 0.232, always-on 0.075).

4. **New concepts.**
   - `piece_mobility(p, sq)`: pseudo-mobility, the squares attacked that are not occupied by the piece's own side.
   - `side_material(p, ours)`: one side's material in pawns.
   - `still_castleable(p, sq)`: the king or rook on sq still holds a castling right.

5. **Caveats.** The strong band and eval context makes it feel like part "low-rated / losing-game" context neuron and part "loose enemy piece" neuron. The empty-square and own-rook firings at band 700 are only roughly captured. Mildly polysemantic. Confidence: medium-low.
