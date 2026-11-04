"""Think-time and outcome-calibration guardrails on human games, per band.

ChessMimic: clock model -> expected think time (bucket centres weighted by probability) vs actual
think time; winner model -> P(black, draw, white). Maia-3: value head (side to move) -> P(...).
Outcome metrics per position after --skip-plies: multiclass Brier, AUC of P(white) - P(black) on
decisive games, and the same restricted to positions where the mover has < 30 s ("clock pressure").

Usage: python -m doppel_lab.guardrails --human eval.parquet --out lab/results/guardrails.json
        [--games-per-band 500] [--device cuda --gpu-ok]
"""

import argparse
import json
import math
import pickle
import sys

import chess
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

from doppel_lab.engines import CM_MODELS, ChessMimic, Maia3, _load_state
from doppel_lab.top1 import positions

PRESSURE_S = 30.0


class Winner:
    def __init__(self, band, device, cc):
        from WinnerTrainer import WinnerPatzerModel
        d = CM_MODELS / "winner_model" / f"{band}_{band + 100}_brier"
        self.s = pickle.load(open(d / "scalers.pkl", "rb"))
        self.model = _load_state(WinnerPatzerModel(
            recent_moves_sequence_length=12, recent_moves_vocab_size=cc.NUM_ACTIONS,
            board_sequence_length=cc.SEQUENCE_LENGTH, board_input_vocab_size=cc.INPUT_VOCAB_SIZE,
            embedding_dim=256, widening_factor=4, num_layers=8, num_heads=8), d / "model.ckpt", device)

    def feats(self, st):
        s = self.s

        def z(v, k):
            return (v - s[k]["mean"]) / s[k]["std"]
        return [z(st.elo_white, "white_rating"), z(st.elo_black, "black_rating"),
                z(math.log(st.clock_white + 1), "log_white_clock"), z(math.log(st.clock_black + 1), "log_black_clock"),
                z(math.log(st.inc + 1), "log_increment")]


def outcome_scores(p, y, pressure):
    """p: (n,3) [black, draw, white]; y: class index; returns Brier and AUC overall and under pressure."""
    out = {}
    for name, m in (("all", np.ones(len(y), bool)), ("pressure", pressure)):
        if m.sum() < 50:
            continue
        pp, yy = p[m], y[m]
        onehot = np.eye(3)[yy]
        dec = yy != 1
        auc = roc_auc_score(yy[dec] == 2, pp[dec, 2] - pp[dec, 0]) if dec.sum() > 10 else float("nan")
        out[name] = {"n": int(m.sum()), "brier": float(((pp - onehot) ** 2).sum(1).mean()), "auc": float(auc)}
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--bands", type=int, nargs="*", default=[1100, 1300, 1500, 1700, 1900, 2100])
    ap.add_argument("--games-per-band", type=int, default=500)
    ap.add_argument("--skip-plies", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--maia-size", default="79m")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-ok", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.device.startswith("cuda") and not args.gpu_ok:
        sys.exit("refusing to touch the GPU without --gpu-ok (run through the warden)")
    dev = torch.device(args.device)
    maia = Maia3(args.maia_size, args.device)
    df = pl.read_parquet(args.human)
    res = {}
    for band in args.bands:
        cm = ChessMimic(band, args.device)
        cc = cm.cc
        win = Winner(band, dev, cc)
        exp_t, act_t, p_cm, p_maia, ys, press = [], [], [], [], [], []
        buf = []

        def flush():
            states = [s for s, _, _, _ in buf]
            recent = torch.tensor([cm._recent(s.moves) for s in states], dtype=torch.long, device=dev)
            board = torch.tensor(np.stack([cc.tokenize(s.board.fen()) for s in states]), dtype=torch.long, device=dev)
            ids = torch.cat([recent, board], 1)
            mask = torch.ones_like(ids)

            def z(v, k):
                return (v - cm.c_scal[k][0]) / cm.c_scal[k][1]
            rat = torch.tensor([z(s.elo_to_move, "rating") for s in states], device=dev)
            f = torch.tensor([[z(math.log(s.clocks[0] + 1), "log_player_clock"),
                               z(math.log(s.clocks[1] + 1), "log_opponent_clock"),
                               z(math.log(s.inc + 1), "log_increment")] for s in states], device=dev)
            pb = torch.softmax(cm.clock_model(ids, mask, rat, f).float(), 1).cpu().numpy()
            centres = np.array([cm.buckets.get_bucket_center(i) for i in range(cm.buckets.n_buckets)])
            exp_t.extend((pb * centres).sum(1).tolist())
            wf = torch.tensor([win.feats(s) for s in states], dtype=torch.float32, device=dev)
            p_cm.extend(torch.softmax(win.model(ids, mask, wf).float(), 1).cpu().numpy().tolist())

            x = torch.stack([maia._input(s) for s in states]).to(maia.device)
            se = torch.tensor([s.elo_to_move for s in states], dtype=torch.long, device=maia.device)
            oe = torch.tensor([s.elo_opp for s in states], dtype=torch.long, device=maia.device)
            with torch.autocast("cuda", enabled=args.device.startswith("cuda")):
                v = torch.softmax(maia.model(x, se, oe)[1].float(), 1).cpu().numpy()  # [loss, draw, win] to move
            for s, (lo, dr, wi) in zip(states, v):
                p_maia.append([lo, dr, wi] if s.board.turn == chess.WHITE else [wi, dr, lo])
            for s, think, y, pr in buf:
                act_t.append(think)
                ys.append(y)
                press.append(pr)
            buf.clear()

        for row in df.filter(pl.col("band") == band).head(args.games_per_band).iter_rows(named=True):
            y = {"0-1": 0, "1/2-1/2": 1, "1-0": 2}.get(row["result"])
            if y is None:
                continue
            clk = row["clocks"]
            for ply, (s, _mv) in enumerate(positions(row, 0, True)):
                if ply < args.skip_plies:
                    continue
                before = s.clocks[0]
                think = before - clk[ply] + s.inc
                buf.append((s, think, y, before < PRESSURE_S))
                if len(buf) >= args.batch:
                    flush()
        if buf:
            flush()
        e, a = np.array(exp_t), np.array(act_t)
        y, pr = np.array(ys), np.array(press)
        res[band] = {
            "positions": len(a),
            "think_time": {"chessmimic_pearson_r": float(np.corrcoef(e, a)[0, 1]),
                           "chessmimic_pearson_r_log": float(np.corrcoef(np.log1p(e), np.log1p(np.maximum(a, 0)))[0, 1])},
            "outcome": {"chessmimic": outcome_scores(np.array(p_cm), y, pr),
                        f"maia3-{args.maia_size}": outcome_scores(np.array(p_maia), y, pr)},
        }
        print(band, json.dumps(res[band]), file=sys.stderr, flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
