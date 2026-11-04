"""Print a compact per-band table from a metrics JSON."""
import json, sys
r = json.load(open(sys.argv[1]))
for b, e in sorted(r["bands"].items()):
    print(f"\n== band {b}")
    print(f"{'source':14}{'games':>6}{'acpl':>7}{'±sd':>6}{'blund':>6}{'len':>5}{'h-corr':>7}{'lift':>6}{'H6':>5}{'cv':>5}{'disp':>5} | KS acpl  KS bl  KS len  endTV  disc  discM")
    for s, v in e.items():
        m = v["summary"]
        d = v.get("vs_human_A")
        row = (f"{s:14}{m['games']:6d}{m['acpl']['mean']:7.1f}{m['acpl']['std']:6.1f}{m['blunders']['mean']:6.2f}"
               f"{m['length']['mean']:5.0f}{m['acpl_half_corr']:7.2f}{m['blunder_lift']:6.2f}{m['open6_entropy_bits']:5.1f}{m['acpl']['cv']:5.2f}{m['blunders']['dispersion']:5.2f}")
        if d:
            row += (f" | {d['acpl']['ks']:7.3f}{d['blunders']['ks']:7.3f}{d['length']['ks']:8.3f}"
                    f"{d['endings_tv']:7.3f}{d['discriminator']['accuracy']:6.2f}{d['discriminator_moves_only']['accuracy']:6.2f}")
        print(row)
        print(" " * 14 + "endings " + ", ".join(f"{k} {x:.2f}" for k, x in sorted(m["endings"].items())))
