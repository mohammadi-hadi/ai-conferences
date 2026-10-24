"""Joint topics for all four sets, so both conferences and both years are measured on the same topics.

k-means on the vectors (40 topics, 10 restarts). Each conference-year gets a share per topic. The overlap of two
conferences is the sum over topics of the smaller of their two shares (1 = same topic mix, 0 = nothing in common),
and the change in that overlap is split by topic, so the story can name the topics that brought them closer.
Writes the material for naming the topics (closest titles and distinctive words) to data/labels/topic_material.md.
"""
import json, re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from config import P, OUT, ROOT, SEED

K = 40
df = pd.read_parquet(P / "papers.parquet")
emb = np.load(P / "embeddings.npy")
km = KMeans(K, n_init=10, random_state=SEED).fit(emb)
df["topic"] = km.labels_
cent = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
df["sim"] = (emb * cent[df["topic"]]).sum(1)
df[["conf", "year", "topic", "sim"]].to_parquet(P / "topics.parquet")
np.save(P / "topic_centroids.npy", cent)

share = df.groupby(["conf", "year"])["topic"].value_counts(normalize=True).unstack(fill_value=0).T
share.columns = [f"{c} {y}" for c, y in share.columns]
ov = {y: np.minimum(share[f"NeurIPS {y}"], share[f"EMNLP {y}"]) for y in (2025, 2026)}
share["shared_2025"], share["shared_2026"] = ov[2025], ov[2026]
share["shared_change"] = ov[2026] - ov[2025]
share["n"] = df["topic"].value_counts()
share.to_csv(OUT / "topic_shares.csv")
summary = {"k": K, "overlap_2025": round(float(ov[2025].sum()), 4), "overlap_2026": round(float(ov[2026].sum()), 4)}
json.dump(summary, open(OUT / "topics.json", "w"), indent=1)
print(summary)

# material for naming
STOP = set("a an the of for and in on to with from by via is are we our this that using based towards toward under "
           "large language models model llm llms learning neural data method methods approach new study through as at "
           "its their be can into".split())
words = lambda t: [w for w in re.findall(r"[a-z][a-z\-]{2,}", t.lower()) if w not in STOP]
glob = Counter(w for t in df["title"] for w in set(words(t)))
lines = []
for k in share.sort_values("n", ascending=False).index:
    d = df[df["topic"] == k]
    c = Counter(w for t in d["title"] for w in set(words(t)))
    dist = sorted(c, key=lambda w: -(c[w] / len(d)) * np.log(len(df) / glob[w]))[:12]
    s = share.loc[k]
    lines.append(f"## topic {k} (n={len(d)}) NeurIPS {s['NeurIPS 2025']:.3f}->{s['NeurIPS 2026']:.3f}, "
                 f"EMNLP {s['EMNLP 2025']:.3f}->{s['EMNLP 2026']:.3f}\nwords: {', '.join(dist)}")
    for _, r in d.nlargest(10, "sim").iterrows():
        lines.append(f"- [{r['conf']} {r['year']}] {r['title']}")
    lines.append("")
(ROOT / "data/labels").mkdir(parents=True, exist_ok=True)
(ROOT / "data/labels/topic_material.md").write_text("\n".join(lines))
print(share.sort_values("shared_change", ascending=False).head(8).round(3).to_string())
