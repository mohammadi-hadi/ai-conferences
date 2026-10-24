"""Portrait images (1080 x 1350) for the LinkedIn post, in outputs/cards/:
1_overlap.png  the hexagon map, 2025 above 2026, with the headline numbers
2_closer.png, 3_topics.png, 4_apart.png  made by 08_figures.py portrait
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from PIL import Image
from config import P, OUT, ROOT

CARDS = OUT / "cards"
CARDS.mkdir(parents=True, exist_ok=True)
INK, MUTED, BG, GREY = "#1f2a44", "#6b7489", "#fbfaf7", "#9aa1b1"
CAT = {"N": "#3b5bdb", "S": "#ae3ec9", "E": "#d9480f"}
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
FOOT = "mohammadi.cv/ai-conferences/2026 · Hadi Mohammadi"

# 1. hexagons, read from the same export as the interactive map
html = (ROOT / "docs/2026/overlap.html").read_text()
data = json.loads(html.split("const DATA = ", 1)[1].split(";\nconst C = ", 1)[0].replace("<\\/", "</"))
w = data["w"]; r = w / np.sqrt(3)
ang = np.deg2rad(np.arange(6) * 60 + 30)
hexagon = np.c_[r * np.cos(ang), r * np.sin(ang)]
xs = np.array([h["x"] for h in data["hexes"]]); ys = np.array([h["y"] for h in data["hexes"]])
fig = plt.figure(figsize=(5.4, 6.75), dpi=200)
fig.patch.set_facecolor(BG)
fig.text(.06, .945, "NeurIPS and EMNLP\nmoved closer", fontsize=22, fontweight="bold", color=INK, va="top", linespacing=1.05)
fig.text(.06, .835, "Papers in areas both conferences share", fontsize=10.5, color=MUTED, va="top")
for i, y in enumerate(("2025", "2026")):
    ax = fig.add_axes([.03 + i * .48, .2, .46, .55])
    ax.set_facecolor(BG)
    cols = [CAT[h[y]["c"]] if h[y]["c"] else ("#e6e8ee" if h[y]["n"] + h[y]["e"] else None) for h in data["hexes"]]
    polys = [hexagon + (x, yy) for x, yy, c in zip(xs, ys, cols) if c]
    ax.add_collection(PolyCollection(polys, facecolors=[c for c in cols if c], edgecolors=BG, linewidths=.3))
    ax.set_xlim(xs.min() - w, xs.max() + w); ax.set_ylim(ys.min() - w, ys.max() + w); ax.set_aspect("equal"); ax.axis("off")
    ax.text(.5, 1.07, y, transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(.5, -.04, f"{data['headline'][y] * 100:.0f}%", transform=ax.transAxes, ha="center", va="top",
            fontsize=30, fontweight="bold", color=CAT["S"])
fig.legend(handles=[Patch(color=CAT["N"], label="Mostly NeurIPS"), Patch(color=CAT["S"], label="Shared"),
                    Patch(color=CAT["E"], label="Mostly EMNLP")], loc="lower center", ncol=3, frameon=False,
           fontsize=9.5, bbox_to_anchor=(.5, .055))
fig.text(.06, .025, FOOT, fontsize=8, color=GREY)
fig.savefig(CARDS / "1_overlap.png", facecolor=BG)
plt.close(fig)

# 2-4: run 08_figures.py portrait, which draws the story figures in the card format
for f in sorted(CARDS.glob("*.png")):
    print(f.name, Image.open(f).size)
