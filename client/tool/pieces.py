"""Generate the Doppel piece set: original minimal silhouettes with a thick sticker outline.

Each piece is a list of filled shapes. Rendering: (1) optional rim stroke, (2) outline stroke of all shapes
(width 2*W, so the union gets one thick silhouette outline with no inner seams), (3) fills, (4) detail marks.
Run: python3 client/tool/pieces.py  -> client/assets/pieces/doppel/{1x..4x}/wK.png etc. (needs inkscape)
"""
import subprocess, sys
from pathlib import Path

W = 5.5  # outline half-width in viewBox units (viewBox 100)
BASE = '<rect x="24" y="78" width="52" height="11" rx="5.5"/>'
SHAPES = {
    "P": ['<circle cx="50" cy="33" r="12.5"/>',
          '<path d="M35 80 C35 62 41 51 50 49 C59 51 65 62 65 80 Z"/>', BASE],
    "R": ['<path d="M27 20 h10 v8 h7.5 v-8 h11 v8 h7.5 v-8 h10 v23 h-46 z"/>',
          '<path d="M32 42 L68 42 L71 80 L29 80 Z"/>', BASE],
    "B": ['<circle cx="50" cy="13.5" r="5.5"/>',
          '<path d="M50 18 C66 31 68 49 60 63 L40 63 C32 49 34 31 50 18 Z"/>',
          '<path d="M38 62 L62 62 L66 80 L34 80 Z"/>', BASE],
    "N": ['<path d="M30 80 C30 63 35 53 44 47 C37 47 30 51 24 49 C19 45 21 38 25 34 '
          'C33 26 40 19 51 16 L55 9 L60 17 C72 21 78 40 73 80 Z"/>', BASE],
    "Q": ['<circle cx="23" cy="31" r="5.5"/>', '<circle cx="50" cy="22" r="5.5"/>', '<circle cx="77" cy="31" r="5.5"/>',
          '<path d="M30 80 L23 34 L38 52 L50 26 L62 52 L77 34 L70 80 Z"/>', BASE],
    "K": ['<rect x="45.5" y="9" width="9" height="26" rx="2"/>', '<rect x="38" y="15" width="24" height="9" rx="2"/>',
          '<path d="M30 80 C27 61 25 47 34 41 C42 36 46 43 50 46 C54 43 58 36 66 41 C75 47 73 61 70 80 Z"/>', BASE],
}
DETAIL = {  # marks drawn on top in the outline colour
    "B": '<path d="M55.5 30 L46 43" stroke-width="5" stroke-linecap="round" fill="none"/>',
    "N": '<circle cx="47" cy="28" r="3.8" stroke="none"/>',
}
STYLE = {
    "w": dict(fill="#F8F4EA", line="#0F1418", rim=None, shade="#E3DCCB"),
    "b": dict(fill="#4A5566", line="#0F1418", rim="#AEBBC7", shade="#3C4555"),
}


def svg(color: str, role: str) -> str:
    s = STYLE[color]
    shapes = "".join(SHAPES[role])
    layers = []
    if s["rim"]:
        layers.append(f'<g fill="{s["rim"]}" stroke="{s["rim"]}" stroke-width="{2 * W + 5}" stroke-linejoin="round">{shapes}</g>')
    layers.append(f'<g fill="{s["line"]}" stroke="{s["line"]}" stroke-width="{2 * W}" stroke-linejoin="round">{shapes}</g>')
    layers.append(f'<g fill="{s["fill"]}">{shapes}</g>')
    # soft lower shade for depth: clip the fills to the bottom band
    layers.append(f'<clipPath id="c"><rect x="0" y="66" width="100" height="40"/></clipPath>'
                  f'<g fill="{s["shade"]}" clip-path="url(#c)">{shapes}</g>')
    if role in DETAIL:
        layers.append(f'<g fill="{s["line"]}" stroke="{s["line"]}">{DETAIL[role]}</g>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1 -1 102 102" width="102" height="102">'
            + "".join(layers) + "</svg>")


def main(out: Path):
    src = out / "svg"
    src.mkdir(parents=True, exist_ok=True)
    for c in "wb":
        for r in SHAPES:
            (src / f"{c}{r}.svg").write_text(svg(c, r))
    for scale, d in ((1, "."), (2, "2.0x"), (3, "3.0x"), (4, "4.0x")):
        (out / d).mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.svg")):
            subprocess.run(["inkscape", str(f), "--export-type=png", f"--export-filename={out / d / (f.stem + '.png')}",
                            f"--export-width={128 * scale}"], check=True, capture_output=True)
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "client/assets/pieces/doppel"))
