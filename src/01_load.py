"""Put the NeurIPS and EMNLP papers of 2025 and 2026 in one table with their vectors.

Duplicates flagged by the atlas pipelines are dropped. Papers without an abstract stay in (their vector is the title).
"""
import json
import numpy as np
import pandas as pd
from config import NEURIPS, EMNLP, P

def load(src, conf, author_sep, title_col):
    df = pd.read_parquet(src / "papers.parquet")
    emb = np.load(src / "embeddings.npy")
    assert len(df) == len(emb)
    model = (src / "embeddings_model.txt").read_text().strip()
    keep = ~df["dup"].to_numpy()
    df = df[keep].reset_index(drop=True)
    out = pd.DataFrame({
        "conf": conf,
        "year": df["year"].astype(int),
        "title": df[title_col],
        "abstract": df["abstract"].fillna(""),
        "authors": df["authors"].fillna("").map(lambda s: [a.strip() for a in s.split(author_sep) if a.strip()]),
        "track": df["track"] if "track" in df else "All",  # the NeurIPS lists do not split tracks
        "url": df["anthology"].fillna(df["pdf"]) if "anthology" in df else df["url"],
    })
    return out, emb[keep], model

n, en, mn = load(NEURIPS, "NeurIPS", ";", "title")
e, ee, me = load(EMNLP, "EMNLP", ",", "title_display")
assert mn == me, (mn, me)
df = pd.concat([n, e], ignore_index=True)
emb = np.vstack([en, ee]).astype(np.float32)
norms = np.linalg.norm(emb, axis=1)
assert np.allclose(norms, 1, atol=1e-2), (norms.min(), norms.max())  # normalised by the model, off by <0.5% (half precision)
emb /= norms[:, None]

df.to_parquet(P / "papers.parquet")
np.save(P / "embeddings.npy", emb)
counts = {}
for (c, y), v in df.groupby(["conf", "year"]).size().items():
    counts.setdefault(c, {})[str(y)] = int(v)
json.dump({"model": mn, "counts": counts},
          open(P / "load.json", "w"), indent=1)
print(mn, df.groupby(["conf", "year"]).size().to_dict())
