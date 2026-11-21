# b8.751 notes

**(1) Description.** The neuron fires on "rook room behind our wing pawns". That means our own rooks, plus the empty squares in our own camp that one of our rooks can reach or that sit behind one of our pawns on the same file. It is strongest on the a/b and g/h files and weaker where the enemy controls the square. The peak case is Ra1 with an empty a2 and a pawn on a3: the rook is boxed in behind an advanced wing pawn.

**(2) Evidence and what I tried.** The top examples (2.5-3.4) are almost all one pattern in the side-to-move frame: an empty square between our corner rook and our own pawn on the same file (a2 with Ra1/Pa3, h2 with Rh1/Ph3, g2, a3 with Ra2/Pa4). Our own rooks fire 44% of the time. Empty back-rank squares fire 22%, falling off steadily with rank. Every other occupied square almost never fires, except our queen at 6%. Among empty squares there are three strong gates:
- an own pawn further up the same file (fire rate 0.05 → 0.2-0.4);
- the square is attacked by our rook;
- the square is on an edge file.

They stack: behind a pawn + rook reach + a/h file fires 58%. Smaller terms:
- enemy attackers on the square (−);
- our own defenders (+);
- an enemy pawn on the file (+);
- our queen's reach (−, oddly).

For rooks, a higher score comes from an enemy pawn on the file, being attacked, being on a wing file, or standing behind our own pawn. I tried several things that did not help:
- Maia/played move from- and to-squares (no signal);
- rook ray direction (file vs rank);
- the "sandwich" gap distance as a firing gate (it only helps magnitude);
- a position-level legal-move-count term (hurt).

The final weights are rounded from a logistic fit on dev empty squares. There are about 10 readable terms for empty squares and 4 for rooks.

**(3) Final dev metrics.** Spearman 0.210, AUROC 0.823, F1 0.427 (threshold −1.65, detector rate 7.96% vs fire rate 7.34%). Baselines: concept pair F1 0.2385, always-on F1 0.137. A gradient-boosted probe on similar features reached about 0.88 AUROC, so the remaining headroom is modest.

**(4) Caveats.**
- Many firings are diffuse and low (0.1-0.5). They are scattered over empty squares, including central squares next to rooks and a few enemy pieces, and the rule does not capture them.
- Rook firing is poorly predicted (AUROC within rooks is about 0.64).
- Spearman is capped because the sub-threshold activations are GELU-negative noise and do not order cleanly: the ceiling is about 0.45 even if firing is predicted perfectly.
- It looks like one fuzzy concept (heavy-piece/rook space on the wings behind our pawns) rather than several unrelated triggers. It is not crisp, though.

Confidence: medium on the peak pattern, low on the broad firing.
