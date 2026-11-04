"""Convert the Allie annotated test set (yimingzhang/allie-data, 2022 blitz) to doppel's game schema.

Clocks are rebuilt from per-move think times: clock_after = clock_before - think + increment.
Band = -1 (the set spans all ratings). Usage: python -m doppel_lab.allie <in.jsonl> <out.parquet>
"""

import json
import sys

import polars as pl


def main():
    rows = []
    for line in open(sys.argv[1]):
        g = json.loads(line)
        base, inc = (int(x) for x in g["time-control"].split("+"))
        clocks, c = [], [float(base), float(base)]
        for i, t in enumerate(g["moves-seconds"]):
            c[i % 2] = c[i % 2] - t + inc
            clocks.append(c[i % 2])
        rows.append({"game_id": g["game-id"].rsplit("/", 1)[-1], "white": "", "black": "",
                     "white_elo": g["white-elo"], "black_elo": g["black-elo"], "band": -1,
                     "base": base, "inc": inc, "result": g["result"], "termination": g["termination"],
                     "moves": g["moves-uci"].split(), "clocks": clocks})
    pl.DataFrame(rows).write_parquet(sys.argv[2], compression="zstd")
    print(f"wrote {len(rows)} games")


if __name__ == "__main__":
    main()
