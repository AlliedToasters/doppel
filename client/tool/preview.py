"""Contact-sheet preview of the Doppel piece set on the Doppel board palette."""
import sys
from pathlib import Path
from PIL import Image

BG, LIGHT = (0x1A, 0x23, 0x29), (0x2C, 0x38, 0x40)
d = Path("client/assets/pieces/doppel/2.0x"); S = 256 // 2
rows = ["rnbqkbnr", "pppppppp", "", "", "", "", "PPPPPPPP", "RNBQKBNR"]
img = Image.new("RGB", (8 * S + 2 * S, 8 * S + 2 * S), BG)
for y in range(8):
    for x in range(8):
        if (x + y) % 2 == 0:
            img.paste(LIGHT, (S + x * S, S + y * S, S + (x + 1) * S, S + (y + 1) * S))
        row = rows[y]
        if row:
            ch = row[x]; name = ("w" if ch.isupper() else "b") + ch.upper()
            p = Image.open(d / f"{name}.png").convert("RGBA").resize((S, S))
            img.paste(p, (S + x * S, S + y * S), p)
img.save(sys.argv[1])
