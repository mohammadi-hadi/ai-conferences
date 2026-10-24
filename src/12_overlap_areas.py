"""Where on the map do the two conferences overlap?

Per year, the joint map is cut into hexagons. Each hexagon gets the EMNLP share of its papers after correcting for
size (each conference's count divided by its total that year), so a hexagon where both conferences have the same share
of their papers scores 0.5. Hexagons below 0.3 are mostly NeurIPS, above 0.7 mostly EMNLP, in between shared.
The share of all papers sitting in shared hexagons is the headline of this figure.

Also writes each paper's neighbourhood mix for the interactive map: of its 20 nearest papers in the same year, the
size-corrected share from the other conference (0 = all its neighbours are from its own conference, 0.5 = even).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 12.5})
from config import P, OUT, FIG
from hexgrid import assign, MIN_N

INK, MUTED, BG, GREY = "#1f2a44", "#6b7489", "#fbfaf7", "#b8bfcc"
CAT = {"Mostly NeurIPS": "#3b5bdb", "Shared": "#ae3ec9", "Mostly EMNLP": "#d9480f"}
SOURCE = "Data: NeurIPS and ACL Anthology paper lists · Analysis: Hadi Mohammadi · github.com/mohammadi-hadi/ai-conferences"

df = pd.concat([pd.read_parquet(P / "papers.parquet"), pd.read_parquet(P / "umap.parquet"),
                pd.read_parquet(P / "topics.parquet")[["topic"]]], axis=1)
emb = np.load(P / "embeddings.npy")

# hexagon id for every paper (one grid for both years, so the areas line up)
x0, x1, y0, y1 = df.x.min(), df.x.max(), df.y.min(), df.y.max()
df["hex"], df["hx"], df["hy"], w = assign(df.x, df.y)
h = w * np.sqrt(3) / 2

out = {}
cells = {}
for y in (2025, 2026):
    d = df[df.year == y]
    tot = d.groupby("conf").size()
    c = d.groupby(["hex", "conf"]).size().unstack(fill_value=0)
    c = c[c.sum(1) >= MIN_N]
    rn, re_ = c["NeurIPS"] / tot["NeurIPS"], c["EMNLP"] / tot["EMNLP"]
    share = re_ / (rn + re_)
    cat = pd.cut(share, [-.01, .3, .7, 1.01], labels=list(CAT))
    cells[y] = pd.DataFrame({"share": share, "cat": cat, "n": c.sum(1)})
    lab = d["hex"].map(cat.astype(str))
    out[str(y)] = {k: round(float((lab == k).mean()), 4) for k in CAT}
    out[str(y)]["papers_in_small_cells"] = round(float(lab.isna().mean()), 4)
    # size-corrected: share of each conference's papers in shared cells
    for conf in ("NeurIPS", "EMNLP"):
        out[str(y)][f"{conf}_papers_in_shared"] = round(float((lab[d.conf == conf] == "Shared").mean()), 4)
# share of each topic's papers that sit in shared hexagons (used in the story and the posts)
names_all = json.load(open(P.parent / "labels/topic_names.json"))
for y in (2025, 2026):
    d = df[df.year == y]
    tot_ = d.groupby("conf").size()
    c = d.groupby(["hex", "conf"]).size().unstack(fill_value=0)
    c = c[c.sum(1) >= MIN_N]
    sh = (c["EMNLP"] / tot_["EMNLP"]) / (c["NeurIPS"] / tot_["NeurIPS"] + c["EMNLP"] / tot_["EMNLP"])
    shared = d["hex"].map((sh > .3) & (sh <= .7)).fillna(False).astype(bool)
    for t, v in shared.groupby(d["topic"]).mean().items():
        out.setdefault("shared_by_topic", {}).setdefault(names_all[str(t)]["name"], {})[str(y)] = round(float(v), 4)
    for a_, v in shared.groupby(d["topic"].map(lambda t: names_all[str(t)]["area"])).mean().items():
        out.setdefault("shared_by_area", {}).setdefault(a_, {})[str(y)] = round(float(v), 4)

# fair check: fewer papers per hexagon makes shares noisier and pushes cells to the extremes, so 2026 is also
# measured on random subsets the size of 2025 (5,856 NeurIPS and 3,484 EMNLP papers, 20 draws)
def shared_share(d):
    tot = d.groupby("conf").size()
    c = d.groupby(["hex", "conf"]).size().unstack(fill_value=0)
    c = c[c.sum(1) >= MIN_N]
    share = (c["EMNLP"] / tot["EMNLP"]) / (c["NeurIPS"] / tot["NeurIPS"] + c["EMNLP"] / tot["EMNLP"])
    cat = pd.cut(share, [-.01, .3, .7, 1.01], labels=list(CAT)).astype(str)
    return float((d["hex"].map(cat) == "Shared").mean())
n25 = df[df.year == 2025].groupby("conf").size()
d26 = df[df.year == 2026]
rng = np.random.default_rng(5)
v = [shared_share(pd.concat([d26[d26.conf == c].sample(int(n25[c]), random_state=int(rng.integers(1e9)))
                             for c in ("NeurIPS", "EMNLP")])) for _ in range(20)]
out["2026_at_2025_size_shared"] = {"mean": round(float(np.mean(v)), 4), "lo": round(float(np.quantile(v, .025)), 4),
                                   "hi": round(float(np.quantile(v, .975)), 4)}
json.dump(out, open(OUT / "overlap_areas.json", "w"), indent=1)
print(json.dumps(out, indent=1))

# neighbourhood mix per paper, same year, size-corrected
mix = np.zeros(len(df))
for y in (2025, 2026):
    ii = np.flatnonzero(df.year.to_numpy() == y)
    X = emb[ii]
    conf = (df.conf.to_numpy()[ii] == "EMNLP").astype(int)
    tot = np.bincount(conf, minlength=2)
    for s in range(0, len(ii), 2000):
        S = X[s:s + 2000] @ X.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -np.inf
        nn = np.argpartition(-S, 20, axis=1)[:, :20]
        ne = conf[nn].sum(1)  # EMNLP neighbours
        re_, rn = ne / tot[1], (20 - ne) / tot[0]
        e_share = re_ / (re_ + rn)
        own = conf[s:s + len(S)]
        mix[ii[s:s + len(S)]] = np.where(own == 1, 1 - e_share, e_share)  # share from the OTHER conference
pd.DataFrame({"neighbour_mix": mix}).to_parquet(P / "neighbour_mix.parquet")
print("neighbour mix, mean by conf-year:", df.assign(m=mix).groupby(["conf", "year"])["m"].mean().round(3).to_dict())

# figure
names = {int(k): v["name"] for k, v in json.load(open(P.parent / "labels/topic_names.json")).items()}
fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
fig.patch.set_facecolor(BG)
r = w / np.sqrt(3)  # pointy-top hexagons that tile the grid exactly
ang = np.deg2rad(np.arange(6) * 60 + 30)
hexagon = np.c_[r * np.cos(ang), r * np.sin(ang)]
for ax, y in zip(axes, (2025, 2026)):
    ax.set_facecolor(BG)
    cc = cells[y].join(df.groupby("hex")[["hx", "hy"]].first())
    alpha = np.clip(.35 + .65 * np.log(cc["n"]) / np.log(cc["n"].quantile(.95)), .35, 1)
    cols = [matplotlib.colors.to_rgba(CAT[k], a) for k, a in zip(cc["cat"].astype(str), alpha)]
    ax.add_collection(PolyCollection([hexagon + (px, py) for px, py in zip(cc["hx"], cc["hy"])],
                                     facecolors=cols, edgecolors=BG, linewidths=.6))
    shared = out[str(y)]["Shared"]
    ax.set_title(f"{y}: {shared * 100:.0f}% of papers in shared areas", fontsize=13.5, color=INK, loc="left",
                 fontweight="bold", pad=4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_xlim(x0 - w, x1 + w); ax.set_ylim(y0 - h, y1 + h)
    for sp in ax.spines.values():
        sp.set_visible(False)
LABELS = {18: "LLM Agents", 31: "RL for LLM\nReasoning", 17: "Retrieval-Augmented\nGeneration",
          8: "Multilingual NLP\nand Translation", 23: "Optimisation\nTheory", 21: "Diffusion and\nFlow Models"}
for t, lab in LABELS.items():
    c = df[df.topic == t][["x", "y"]].median()
    for ax in axes:
        ax.text(c.x, c.y, lab, fontsize=10, fontweight="bold", color=INK, ha="center", va="center", multialignment="center",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
axes[0].text(0, 1.12, "Where the two conferences overlap", fontsize=17, fontweight="bold", color=INK, transform=axes[0].transAxes)
axes[0].text(0, 1.065, "Each hexagon is a patch of the joint map, coloured by which conference its papers come from. "
             "Darker hexagons have more papers.", fontsize=11, color=MUTED, transform=axes[0].transAxes)
fig.legend(handles=[Patch(color=v, label=k) for k, v in CAT.items()], loc="lower center", ncol=3, frameon=False,
           fontsize=12, bbox_to_anchor=(.5, .02))
fig.text(0.01, 0.0, SOURCE, fontsize=8, color=GREY, transform=fig.transFigure, va="top")
fig.savefig(FIG / "overlap_areas.png", dpi=200, bbox_inches="tight", facecolor=BG)
print("wrote overlap_areas.png")
