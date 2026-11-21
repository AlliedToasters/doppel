# b5.633 notes

**(1) Description.** This is a diffuse "active square for the side to move" neuron. It fires on squares the mover's queen, bishop, rook or king has a hand on, especially when the opponent also contests the square, an enemy piece sits there, or it is where the mover is likely to go: Maia's top destinations, Stockfish's move or the opponent's best reply. It fires more often in stronger rating bands and in endgames.

**(2) Evidence and what I tried.** The neuron fires on only 0.72% of dev tokens, usually on 1 or 2 squares per position, and 32% of positions have at least one firing square.

- Coverage by our own pieces is the strongest single signal. Squares with no non-pawn cover fire at 0.19%. Fire rate rises with the number of covering pieces: 1.2% for one, up to 2.8% for four. Queen and bishop cover count most, and pawn cover counts against.
- Other enrichments, each worth about 2-5x:
  - our king's square and the squares next to it;
  - contested squares, where we cover with a piece and they attack;
  - squares with an enemy piece on them;
  - legal-move targets and Maia top-5 destinations (castling counts the rook's square too);
  - the squares the SF PV's first two moves land on.
- At position level, the rating band matters: 18% of 700-band positions have a firing square, against about 38% for 1500 and above.
- Closer reading of the top examples turned up no sharper rule. Many show a queen deep in enemy territory lighting a nearby empty square, castling squares d1/f1, or king-walk squares in K+Q endgames, but no single trigger. Breaking cover down by attacker type and rank found nothing sharp either.

The final rule is an additive score of about 12 hand-set weights, rounded from a logistic fit. For comparison, a gradient-boosted model on about 40 board features reached only AUROC 0.86 / F1 0.156 on a held-out half of dev. That is roughly the ceiling for board-level features.

**(3) Final dev metrics.** Spearman -0.314, AUROC 0.842, F1 0.120. The best python-chess concept pair scored F1 0.081.

**(4) Caveats.**
- **Spearman is negative because of the GELU shape, not because the rule is wrong.** Squares the rule rates "almost firing" sit in the GELU dip near -0.17. Squares it rates "far off" sit near 0. So within the 99% of non-firing tokens, raw activation runs against the score. A GELU-shaped output can push Spearman to about +0.28, but only by dropping AUROC below 0.5, so I kept the output aimed at firing.
- Precision is low: even the top-scoring 5% of tokens fire only about 5% of the time. Heavily covered squares around our own king are the main false positives, for example recapture squares like Qxd7/Nxd7 where the neuron stays silent.
- It misses lone-piece squares that look uncontested, such as a queen on g7 lighting h8.
- It looks polysemantic, or depends on context that is not visible from simple board features.
- Confidence: low.

Analysis scripts in this folder: analyze.py, feats2.py, feats3.py, qchk.py, evalbins.py (dev rows only).
