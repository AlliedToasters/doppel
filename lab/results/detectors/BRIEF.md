# Brief: write an executable explanation of one Maia-3 neuron

Maia-3-79M predicts human chess moves. It is an 8-block transformer whose 64 tokens are the board squares, seen
from the side to move (the board is mirrored when black is to move). Your job: explain ONE neuron of its MLP
(GELU activation of linear1) by writing `detect(board, square, info) -> float` in python, which should be high on
the (position, square) tokens where the neuron fires and low elsewhere. The explanation is the code.

Working dir: /home/alliedtoasters/projects/chess/bot  (run python via `uv run --project lab python ...`)
Your files: data/allie/neuron_atlas/detect/<NEURON>/
  evidence.md   top-activating examples, weak firings, silent squares, with boards, Maia-3 and Stockfish lines
  detector.py   template; edit it (keep the signature)
  notes.md      write this at the end (see below)

Score on the dev split (allowed as often as you like, ~1-30 s each):
  uv run --project lab python -m doppel_lab.detectors score --neuron <NEURON> --detector data/allie/neuron_atlas/detect/<NEURON>/detector.py
It prints dev Spearman (detector vs activation over all tokens), AUROC for firing (activation > 0.1) and F1 at the
best threshold. An always-on detector gets F1 = 2p/(1+p) for fire rate p; the best python-chess concept pair is
given in evidence.md and is the baseline to beat.

You may also analyse the dev data directly with your own scripts (put them in your neuron folder):
  data/allie/neuron_atlas/detect/evalset/meta.parquet  (polars; columns fen, played, maia_top5, maia_p5, sf_cp,
      sf_pv, ply, band, split) - use ONLY rows with split == "dev"
  data/allie/neuron_atlas/detect/evalset/<NEURON>.npy  activations (positions, 64 tokens); token index is the square
      in Maia-3's mirrored frame: real square = chess.square_mirror(token) when black is to move, else token.
  Rows of the .npy align with meta rows. NEVER read or score rows with split == "test"; they are the held-out set.

Concept library: lab/doppel_lab/chess_concepts.py (`Pos.of(board, info)`: side-to-move frame, who guards a square,
"guarded only by", x-rays and lines, pawn structure by file, Maia move-target mass, Stockfish PV, rating / phase /
eval context, `gelu` output). Read it first and build on it; worked example: lab/results/detectors/b5.210/detector_lib.py.
If you need a reusable concept it lacks, write it in your detector as a small function with a docstring, and list it
under "new concepts" in notes.md (do not edit the library; the orchestrator merges them).

Rules
- python-chess, doppel_lab.chess_concepts and the `info` dict only inside detect() (no engine calls; it runs on ~380k tokens).
- Prefer an explanation a chess player could read: a clear rule (maybe with a few graded terms) beats an opaque fit.
  No lookup tables keyed on exact FENs, no fitted models with more than ~10 parameters.
- Iterate: hypothesis -> code -> score -> look at the failures (false positives and misses on dev) -> refine.
  Stop when improvements flatten, or after about 12 scoring runs.
- notes.md: (1) one-sentence description of what the neuron detects in chess terms, (2) a short paragraph of the
  evidence and what you tried, (3) final dev metrics, (4) new concepts (reusable helpers you wrote, one line each), (5) honest caveats: what it misses, whether it looks
  polysemantic (several unrelated triggers), confidence low/medium/high.
- Final reply to the orchestrator: the one-sentence description, final dev Spearman / AUROC / F1, and confidence.
  Under 120 words.
