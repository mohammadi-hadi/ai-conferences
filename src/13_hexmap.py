"""Interactive overlap map (docs/2026/overlap.html): the hexagons of 12_overlap_areas.py, with a year switch and
several views (overlap, gradient, density, change, papers). The data is exported here from the same grid as the
static figure, so the page and the figure always agree; the page itself is src/hexmap_template.html.
"""
import json
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
from config import P, OUT, ROOT
from hexgrid import assign, category, MIN_N

df = pd.concat([pd.read_parquet(P / "papers.parquet"), pd.read_parquet(P / "umap.parquet"),
                pd.read_parquet(P / "topics.parquet")[["topic", "sim"]]], axis=1)
names = {int(k): v["name"] for k, v in json.load(open(ROOT / "data/labels/topic_names.json")).items()}
x0, x1, y0, y1 = df.x.min(), df.x.max(), df.y.min(), df.y.max()
df["hex"], df["hx"], df["hy"], w = assign(df.x, df.y)
h = w * np.sqrt(3) / 2

cat = category

tot = df.groupby(["year", "conf"]).size()
# a few NeurIPS papers share one session link; send those to a search for the title instead
dup = df["url"].duplicated(keep=False)
df.loc[dup, "url"] = "https://www.google.com/search?q=" + df.loc[dup, "title"].str.strip().map(quote_plus)
hexes = []
for hid, g in df.groupby("hex"):
    rec = {"id": int(hid), "x": round(float(g.hx.iloc[0]), 4), "y": round(float(g.hy.iloc[0]), 4)}
    for y in (2025, 2026):
        d = g[g.year == y]
        n, e = int((d.conf == "NeurIPS").sum()), int((d.conf == "EMNLP").sum())
        share = None
        if n + e >= MIN_N:
            rn, re_ = n / tot[(y, "NeurIPS")], e / tot[(y, "EMNLP")]
            share = round(float(re_ / (rn + re_)), 4)
        top = [names[t] for t in d.topic.value_counts().index[:3]]
        # a few papers to show, from both conferences, the most typical of their topic first
        papers = []
        for conf in ("NeurIPS", "EMNLP"):
            for _, r in d[d.conf == conf].nlargest(6, "sim").iterrows():
                papers.append([r.title.strip(), r.url, conf[0]])
        rec[str(y)] = {"n": n, "e": e, "s": share, "c": cat(share), "t": top, "p": papers}
    hexes.append(rec)

# check: the shares of papers in shared hexagons must match the static figure
fig = json.load(open(OUT / "overlap_areas.json"))
for y in ("2025", "2026"):
    shared = sum(r[y]["n"] + r[y]["e"] for r in hexes if r[y]["c"] == "S") / sum(r[y]["n"] + r[y]["e"] for r in hexes)
    assert abs(shared - fig[y]["Shared"]) < 1e-3, (y, shared, fig[y]["Shared"])

LABELS = {18: "LLM Agents", 31: "RL for LLM Reasoning", 17: "RAG",
          8: "Multilingual NLP and Translation", 23: "Optimisation Theory", 21: "Diffusion and Flow Models",
          13: "Video Understanding", 33: "Molecules and Proteins"}
labels = [{"t": lab, "x": round(float(df[df.topic == t].x.median()), 3), "y": round(float(df[df.topic == t].y.median()), 3)}
          for t, lab in LABELS.items()]
q = lambda v: (np.round((v - v.min()) / (v.max() - v.min()) * 9999)).astype(int).tolist()
dots = {"x": q(df.x), "y": q(df.y), "c": (df.conf == "EMNLP").astype(int).tolist(), "yr": (df.year == 2026).astype(int).tolist(),
        "x0": float(x0), "x1": float(x1), "y0": float(y0), "y1": float(y1)}
counts = {c: {str(y): int(tot[(y, c)]) for y in (2025, 2026)} for c in ("NeurIPS", "EMNLP")}
data = {"w": float(w), "hexes": hexes, "labels": labels, "dots": dots, "counts": counts,
        "headline": {"2025": fig["2025"]["Shared"], "2026": fig["2026"]["Shared"],
                     "2026_at_2025_size": fig["2026_at_2025_size_shared"]["mean"]}}
blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
page = (ROOT / "src/hexmap_template.html").read_text().replace("__DATA__", blob)
(ROOT / "docs/2026").mkdir(parents=True, exist_ok=True)
(ROOT / "docs/2026/overlap.html").write_text(page)
print(len(hexes), "hexagons,", round(len(blob) / 1e6, 2), "MB of data")
