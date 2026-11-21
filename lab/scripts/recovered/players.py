import polars as pl, sys
c=pl.col
def stats(name, path, w="white", b="black"):
    lf=pl.scan_parquet(path)
    s=pl.concat([lf.select(p=c(w).str.to_lowercase()), lf.select(p=c(b).str.to_lowercase())]).drop_nulls()
    g=s.group_by("p").len().collect()["len"]
    print(f"{name:28s} games {int(g.sum())//2:>11,}  players {g.len():>9,}  median {g.median():>5.0f}  q90 {g.quantile(.9):>6.0f}  40+ {(g>=40).sum():>8,}", flush=True)
for a in sys.argv[1:]:
    n,p=a.split("=",1); stats(n,p)
