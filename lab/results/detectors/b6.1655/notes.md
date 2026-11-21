# b6.1655 notes

**(1) Description.** A strong-player neuron: in high-rated games (2300 band) it marks enemy minor pieces and pawns
that have pushed into our half, and squares that only our queen or king touch or that the queens cross-fire,
provided no enemy pawn guards the square.

**(2) Evidence and what I tried.** Rating band is the biggest single factor. The fire rate is 3.4% of tokens at 2300,
0.8% at 1900 and about 0.2% at 700 to 1500, and 78% of 2300 positions have at least one firing square. Within the 2300
band, enemy-occupied squares fire 5x more often than others (knights 15%, bishops 22%, pawns 10%; enemy rooks, queens
and kings about 3%). The rate grows the deeper the enemy piece stands (30-36% on our 2nd/3rd rank, 2% on their own back
rank) and drops when it is next to its own king. Other lifts: squares whose cheapest attacker on our side is the
queen or king (6% empty, 28% enemy-occupied), squares the enemy queen attacks, and squares on an enemy slider line aimed
at our king or queen. Suppressors: an enemy pawn guarding the square, our pawns covering it, our own rooks, and both
back ranks. I also checked Maia/SF move squares (top-move destination lifts 3.5x but adds nothing once these terms are
in), pins, x-ray rays, mobility and position-level features (material, ply, eval, entropy; all |rho| < 0.16). None of
them added much. The detector is a hand-rounded log-odds sum of about 14 readable terms, with weights taken from a
logistic fit on the 2300 dev band.

**(3) Final dev metrics.** Spearman -0.319, AUROC 0.888, F1 0.237 (threshold 2.37; baseline concept pair F1 0.085;
always-on F1 0.018).

**(4) Caveats.** The negative Spearman is structural, not a bug. Most tokens sit on GELU's negative branch, where a more
strongly suppressed pre-activation gives an activation closer to 0 (e.g. our own rooks, about -0.02) than a
nearly-firing token (about -0.17). So any score that ranks the firing candidates highly is anti-correlated across
the silent bulk. Passing the score through a GELU raises Spearman to about +0.3 but drops AUROC to about 0.45, so I kept
the monotone score for detection. Precision is still low: within a position the neuron picks 2-3 of many similar
candidate squares by context I could not pin down. The firing squares often cluster around a tactical hotspot,
e.g. the squares around a king under attack, or the target of a forcing queen or king move. It looks diffuse or
polysemantic ("tension / intrusion squares in strong-player games") rather than one crisp concept.
Confidence: low.
