# b6.890 notes

**(1) Description.** It fires on empty central squares that sit below all of the side-to-move's own pawns on
that file (an open or half-open file, or the square behind an advanced own pawn, such as d2/d3 after the d-pawn
has moved). Activation is strongest when our knights, bishops or queen cover the square, and when the square
is directly in front of an enemy pawn (a6 against a7 on a half-open a-file, c3 against c4).

**(2) Evidence and what I tried.** On dev, the fire rate on empty squares is about 7-10% when no own pawn sits
below the square on its file, against about 1-2% when one does. Own pawns, rooks, queen and king on the square
almost never fire (0.0-0.3%). Coverage by our minor pieces or queen raises the rate roughly 3x per piece. Rook
coverage does nothing. The fire-rate map peaks on relative d2/d3/e2/e3/c3 and falls away towards the rim files
and the back rank. In the files with no own pawn, the square directly below the nearest enemy pawn fires 23%
of the time, against 6-12% for other squares on that file.

Hypotheses I rejected:
- Destinations of Maia's or Stockfish's moves, squares newly attacked by Maia's top moves, and move paths.
  Each scored AUROC ≤ 0.63.
- Enemy-threat squares.
- Rook-line reach.
- Position-level effects (eval, ply, material). These matter little: GBM gain ≤ 0.015.

The detector is a hand-readable additive score with 9 terms, weighted by a logistic fit on dev firing:
- empty square
- own pawn, rook, queen or king on the square (switches it off)
- number of our N/B/Q attackers
- number of our pawn attackers
- no own pawn below the square on its file
- blockade square directly in front of an enemy pawn
- attacked by an enemy pawn
- file centrality (graded)
- not our back rank

As a reference ceiling, a gradient-boosted tree on about 30 board-local features reaches AUROC about 0.89
(0.90 with eval and ply added).

**(3) Final dev metrics** (7 scoring runs):
- spearman_pos 0.269
- AUROC 0.849
- F1 0.299 at threshold -2.45
- Plain Spearman -0.227

Baselines are F1 0.171 for the best concept pair and 0.066 for always-on.

Plain Spearman is negative because the score is monotone while GELU is not. Strongly rejected squares (own pawns,
pieces) have activations of about 0, which rank above the mildly-off squares at about -0.15. Passing the score
through a GELU gives Spearman about +0.27, but AUROC falls to 0.5-0.7. I kept the monotone score;
spearman_pos is the relevant number.

**(4) Caveats.**
- Precision is low. Many d/e-file squares that are central, empty, file-clear and minor-covered do not fire,
  and nothing in the board features I tried separates them. A tree model leaves about 0.04 AUROC of interactions
  on the table.
- It misses scattered firings that look tactical or line-based, such as squares on a rank a rook is about to
  use, squares near an enemy queen or bishop incursion, and promotion squares (d8 in front of a passed d7 pawn).
- The neuron looks like one broad "rook-accessible / file-structure entry square" feature with a positional
  prior, rather than several unrelated triggers. However, the long tail of firings is not explained.
- Confidence: low-medium.
