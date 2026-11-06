"""python -m doppel_lab.analyzer <game> [--out game.html] [--tables DIR] [--no-maia] [--no-stockfish] [--nodes N]"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from doppel_lab.analyzer import render, source
from doppel_lab.features import channels, transforms


def analyze(spec: str, maia: bool = True, stockfish: bool = True, nodes: int = 50_000,
            stockfish_path: str | None = None) -> dict:
    rec = source.validate(source.load(spec))
    pos, mov, boards = channels.base_channels(rec)
    timings = {}
    if maia:
        t = time.time()
        channels.maia_channels(rec, pos, mov, boards)
        timings["maia_s"] = round(time.time() - t, 1)
    if stockfish:
        path = stockfish_path or shutil.which("stockfish") or "/usr/games/stockfish"
        t = time.time()
        channels.stockfish_channels(pos, mov, boards, nodes=nodes, path=path)
        timings["stockfish_s"] = round(time.time() - t, 1)
    side_tf = transforms.side_transforms(mov)
    pos_tf = transforms.position_transforms(pos)
    evs = transforms.events(rec, pos, mov)
    summ = transforms.summary(rec, pos, mov, pos_tf, evs)
    return {"rec": rec, "pos": pos, "mov": mov, "side_tf": side_tf, "pos_tf": pos_tf, "events": evs,
            "summary": summ, "timings": timings}


def write_tables(a: dict, out: Path):
    """Per-ply and per-position tables plus the per-side summary, for estimators built on top."""
    import polars as pl

    out.mkdir(parents=True, exist_ok=True)
    gid = a["rec"]["game_id"]
    mov = {k: v for k, v in (a["mov"] | a["side_tf"]).items()}
    mov["cls"] = [str(c) for c in mov["cls"]] if "cls" in mov else None
    pl.DataFrame({k: list(v) for k, v in mov.items() if v is not None}).write_parquet(out / f"{gid}_moves.parquet")
    pl.DataFrame({k: list(v) for k, v in (a["pos"] | a["pos_tf"]).items()}).write_parquet(out / f"{gid}_positions.parquet")
    pl.DataFrame(a["summary"]).write_parquet(out / f"{gid}_summary.parquet")
    (out / f"{gid}_events.json").write_text(json.dumps(a["events"], indent=1))


def main():
    ap = argparse.ArgumentParser(prog="python -m doppel_lab.analyzer", description=__doc__)
    ap.add_argument("game", help="Lichess game ID or URL (finished games only), a .pgn file, or a v0 game_id")
    ap.add_argument("--out", help="HTML path (default: <game_id>.html in the current directory)")
    ap.add_argument("--tables", help="also write per-ply parquet tables and the summary to this directory")
    ap.add_argument("--no-maia", action="store_true", help="skip Maia-3 (T1)")
    ap.add_argument("--no-stockfish", action="store_true", help="skip Stockfish (T2)")
    ap.add_argument("--nodes", type=int, default=50_000, help="Stockfish nodes per position (default 50000)")
    ap.add_argument("--stockfish", help="path to the Stockfish binary")
    a = ap.parse_args()
    try:
        res = analyze(a.game, maia=not a.no_maia, stockfish=not a.no_stockfish, nodes=a.nodes,
                      stockfish_path=a.stockfish)
    except source.LiveGame as e:
        sys.exit(str(e))
    out = Path(a.out or f"{res['rec']['game_id']}.html")
    shown = {k: v for k, v in res["side_tf"].items() if not k.endswith("_prev")}  # lags duplicate the plotted series
    out.write_text(render.render(res["rec"], res["pos"], res["mov"], shown, res["pos_tf"], res["events"],
                                 res["summary"]))
    if a.tables:
        write_tables(res, Path(a.tables))
    n = len(res["mov"]["ply"])
    print(f"{res['rec']['game_id']}: {n} plies, {len(res['events'])} events, {res['timings']} -> {out}")


if __name__ == "__main__":
    main()
