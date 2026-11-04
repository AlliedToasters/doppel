"""Player-ID split. A player is held out iff sha1(name) % 5 == 0 (about 20% of players).

Training must drop every game with a held-out player; evaluation uses only games with at least one.
`python -m doppel_lab.split <games.parquet> <eval.parquet> [--per-band 2000]` writes the human eval set.
"""

import argparse
import hashlib

import polars as pl


def held_out(name: str) -> bool:
    return int(hashlib.sha1(name.lower().encode()).hexdigest(), 16) % 5 == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--per-band", type=int, default=2000)
    args = ap.parse_args()
    df = pl.read_parquet(args.src)
    test = df.filter(pl.col("white").map_elements(held_out, return_dtype=pl.Boolean)
                     | pl.col("black").map_elements(held_out, return_dtype=pl.Boolean))
    out = test.group_by("band", maintain_order=True).head(args.per_band).with_columns(source=pl.lit("human"))
    out.write_parquet(args.out, compression="zstd")
    print(f"{len(test)} of {len(df)} games touch a held-out player; wrote", out.group_by("band").len().sort("band"))


if __name__ == "__main__":
    main()
