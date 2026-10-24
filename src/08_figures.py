"""Story figures, in the style of the two atlases. Titles state what each chart shows and are filled in from the numbers.

outputs/figures/:
- closeness.png       share of papers whose closest paper is at the other conference, 2025 vs 2026
- topics_together.png the topics behind the gain in shared ground, share per conference, 2025 -> 2026
- map_years.png       the joint map, 2025 and 2026 side by side, NeurIPS and EMNLP in two colours
- still_apart.png     the topics that are still mostly one conference's, 2026
"""
import json, sys
import numpy as np
import pandas as pd
import matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from config import P, OUT, FIG, ROOT

INK, MUTED, BG, GREY = "#1f2a44", "#6b7489", "#fbfaf7", "#b8bfcc"
COL = {"NeurIPS": "#3b5bdb", "EMNLP": "#d9480f"}
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 12.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9ced8",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
                     "legend.labelcolor": INK, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "savefig.bbox": "tight", "savefig.dpi": 200})
PORTRAIT = len(sys.argv) > 1 and sys.argv[1] == "portrait"  # 1080 x 1350 versions for the LinkedIn post
CARD = {"closeness.png": "2_closer.png", "topics_together.png": "3_topics.png", "still_apart.png": "4_apart.png"}
FS = lambda size: (5.4, 6.75) if PORTRAIT else size
SOURCE = "Data: NeurIPS and ACL Anthology paper lists · Analysis: Hadi Mohammadi · github.com/mohammadi-hadi/ai-conferences"

def finish(fig, ax, title, sub, name, src_y=-0.04):
    fig.text(0.0, 1.08, title, fontsize=16, fontweight="bold", color=INK, transform=ax.transAxes, ha="left")
    fig.text(0.0, 1.025, sub, fontsize=10.5, color=MUTED, transform=ax.transAxes, ha="left")
    fig.text(0.01, src_y, SOURCE, fontsize=8, color=GREY, transform=fig.transFigure, ha="left", va="top")
    if PORTRAIT:
        (OUT / "cards").mkdir(exist_ok=True)
        left = {"closeness.png": .26, "topics_together.png": .52, "still_apart.png": .62}[name]
        fig.subplots_adjust(left=left, right=.94, top=.8, bottom=.17)
        for t in fig.texts:  # the source line goes inside the card
            if t.get_text() == SOURCE:
                t.set_transform(fig.transFigure); t.set_position((.03, .025)); t.set_fontsize(6.6); t.set_color(MUTED); t.set_va("bottom")
        for t in fig.texts:  # title and subtitle start at the left edge of the card
            if t.get_text() != SOURCE:
                t.set_transform(fig.transFigure); t.set_x(.04)
                t.set_y(.93 if t.get_fontsize() >= 15 else .885)
        with plt.rc_context({"savefig.bbox": "standard"}):  # keep the exact 1080 x 1350 frame
            fig.savefig(OUT / "cards" / CARD[name], dpi=200)
    else:
        fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)

D = json.load(open(OUT / "direction.json"))["pairs"]
names = {int(k): v["name"] for k, v in json.load(open(ROOT / "data/labels/topic_names.json")).items()}
sh = pd.read_csv(OUT / "topic_shares.csv", index_col=0)

# 1. closeness
fig, ax = plt.subplots(figsize=FS((6.4, 3.4)))
rows = [("NeurIPS papers", "neurips_closest_at_emnlp", "NeurIPS"),
        ("EMNLP papers", "emnlp_closest_at_neurips", "EMNLP")]
for i, (lab, key, conf) in enumerate(rows):
    a, b = D["NeurIPS 2025 vs EMNLP 2025"][key], D["NeurIPS 2026 vs EMNLP 2026"][key]
    yy = len(rows) - 1 - i
    ax.annotate("", xy=(b["mean"] * 100, yy - .22), xytext=(a["mean"] * 100, yy - .22),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4, mutation_scale=11))
    for v, yr, alpha in ((a, 2025, .45), (b, 2026, 1)):
        ax.errorbar(v["mean"] * 100, yy, xerr=[[100 * (v["mean"] - v["lo"])], [100 * (v["hi"] - v["mean"])]],
                    fmt="o", ms=11, color=COL[conf], alpha=alpha, capsize=4, lw=1.5, zorder=2)
        ax.annotate(f"{yr}: {v['mean'] * 100:.0f}%", (v["mean"] * 100, yy), xytext=(-6 if yr == 2025 else 6, 13),
                    textcoords="offset points", ha="right" if yr == 2025 else "left", fontsize=10, color=INK)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlim(15, 35)
ax.set_xticks(range(15, 36, 5))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylim(-.6, 1.7)
ax.set_xlabel("Closest paper at the other conference\n(3,000 papers per side; bars: spread over 50 draws)")
n25, n26 = D["NeurIPS 2025 vs EMNLP 2025"]["neurips_closest_at_emnlp"]["mean"], D["NeurIPS 2026 vs EMNLP 2026"]["neurips_closest_at_emnlp"]["mean"]
finish(fig, ax, "Closer than a year ago",
       f"NeurIPS papers with their closest paper at EMNLP: {n25 * 100:.0f}% → {n26 * 100:.0f}%",
       "closeness.png", src_y=-0.13)

# 2. topics behind the gain
top = sh.sort_values("shared_change", ascending=False).head(6)
fig, ax = plt.subplots(figsize=FS((6.8, 5.0)))
for i, (t, r) in enumerate(top.iterrows()):
    y = len(top) - 1 - i
    for conf, dy in (("NeurIPS", .17), ("EMNLP", -.17)):
        a, b = 100 * r[f"{conf} 2025"], 100 * r[f"{conf} 2026"]
        if abs(b - a) < .3:  # no visible change: a dot, not a stub of an arrow
            ax.plot(b, y + dy, "o", color=COL[conf], ms=6)
            continue
        ax.annotate("", xy=(b, y + dy), xytext=(a, y + dy),
                    arrowprops=dict(arrowstyle="-|>", color=COL[conf], lw=2.2, mutation_scale=13))
        ax.plot(a, y + dy, "o", color=COL[conf], ms=5, alpha=.5)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([names[int(t)] for t in top.index][::-1])
ax.set_xlabel("Share of papers, 2025 → 2026")
ax.set_xlim(0, None)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
for conf, y in (("NeurIPS", .14), ("EMNLP", .07)):
    ax.text(.97, y, f"● {conf}", color=COL[conf], fontsize=11, fontweight="bold", transform=ax.transAxes, ha="right")
T = json.load(open(OUT / "topics.json"))
finish(fig, ax, "Agents and RL gave the biggest gains",
       f"Topics where common ground grew most. Overall: {T['overlap_2025'] * 100:.0f}% → {T['overlap_2026'] * 100:.0f}%",
       "topics_together.png")

if not PORTRAIT:
    # 3. joint map, two years
    df = pd.read_parquet(P / "papers.parquet")
    xy = pd.read_parquet(P / "umap.parquet")
    df = pd.concat([df, xy], axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, yr in zip(axes, (2025, 2026)):
        d = df[df["year"] == yr]
        for conf in ("NeurIPS", "EMNLP"):
            e = d[d["conf"] == conf]
            ax.scatter(e["x"], e["y"], s=1.4, color=COL[conf], alpha=.35, linewidths=0, rasterized=True)
        ax.set_title(f"{yr}: {len(d[d.conf == 'NeurIPS']):,} NeurIPS and {len(d[d.conf == 'EMNLP']):,} EMNLP papers",
                     fontsize=11.5, color=INK, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(df["x"].min() - .2, df["x"].max() + .2); ax.set_ylim(df["y"].min() - .2, df["y"].max() + .2)
        for s in ax.spines.values():
            s.set_visible(False)
    topics = pd.read_parquet(P / "topics.parquet")
    df["topic"] = topics["topic"]
    # labels placed by hand at each topic's median position, wrapped so they do not run into each other
    LABELS = {18: "LLM Agents", 31: "RL for LLM\nReasoning", 17: "Retrieval-Augmented\nGeneration",
              8: "Multilingual NLP\nand Translation", 23: "Optimisation\nTheory", 21: "Diffusion and\nFlow Models"}
    for t, lab in LABELS.items():
        c = df[df["topic"] == t][["x", "y"]].median()
        for ax in axes:
            ax.text(c["x"], c["y"], lab, fontsize=10, fontweight="bold", color=INK, ha="center", va="center",
                    multialignment="center", path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    axes[0].text(0, 1.13, "The same map, a year apart", fontsize=16, fontweight="bold", color=INK, transform=axes[0].transAxes)
    axes[0].text(0, 1.075, "Each dot is a paper; papers on similar topics sit close together. "
                 "Blue: NeurIPS. Orange: EMNLP.", fontsize=10.5, color=MUTED, transform=axes[0].transAxes)
    fig.text(0.01, -0.01, SOURCE, fontsize=8, color=GREY, transform=fig.transFigure, va="top")
    fig.savefig(FIG / "map_years.png")
    plt.close(fig)
    print("wrote map_years.png")

# 4. still apart, 2026 (topics with at least 1% of either conference)
s = sh[(sh["NeurIPS 2026"] >= .01) | (sh["EMNLP 2026"] >= .01)].copy()
s["lean"] = np.log2((s["EMNLP 2026"] + 1e-3) / (s["NeurIPS 2026"] + 1e-3))
pick = pd.concat([s.nsmallest(5, "lean"), s.nlargest(5, "lean")[::-1]])
fig, ax = plt.subplots(figsize=FS((6.8, 5.4)))
ys = np.arange(len(pick))[::-1]
for y, (t, r) in zip(ys, pick.iterrows()):
    ax.barh(y + .18, 100 * r["NeurIPS 2026"], height=.34, color=COL["NeurIPS"])
    ax.barh(y - .18, 100 * r["EMNLP 2026"], height=.34, color=COL["EMNLP"])
ax.set_yticks(ys)
ax.set_yticklabels([names[int(t)] for t in pick.index])
ax.set_xlabel("Share of papers in 2026")
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
for conf, y in (("NeurIPS", .95), ("EMNLP", .88)):
    ax.text(.97, y, f"■ {conf}", color=COL[conf], fontsize=11, fontweight="bold", transform=ax.transAxes, ha="right")
finish(fig, ax, "What still sets them apart",
       "The five topics most specific to each conference in 2026", "still_apart.png")
