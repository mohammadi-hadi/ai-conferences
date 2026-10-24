"""A second lens that does not depend on vectors: do the same people publish at both conferences?

Names are matched loosely (lower case, accents and punctuation removed, first name + last name only), so two
different people with the same name count as one. That inflates both years about equally, so the change matters
more than the level. Reports, per year, the share of EMNLP papers with at least one author who also has a NeurIPS
paper that year, the reverse, and the share of distinct authors at both.

Bigger conferences share more authors by chance alone, and NeurIPS grew 56%. So the 2026 numbers are also computed on
random 2026 subsets the size of 2025 (5,856 NeurIPS and 3,484 EMNLP papers, 50 draws): that is the fair comparison.
"""
import numpy as np
import json, re, unicodedata
import pandas as pd
from config import P, OUT

df = pd.read_parquet(P / "papers.parquet")

def norm(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    parts = [p for p in re.sub(r"[^a-z\s\-]", " ", s).replace("-", " ").split() if len(p) > 1 or p.isalpha()]
    parts = [p for p in parts if len(p) > 1]
    return f"{parts[0]} {parts[-1]}" if len(parts) >= 2 else (parts[0] if parts else "")

df["names"] = df["authors"].map(lambda a: {norm(x) for x in a} - {""})
def lens(d):
    sets = {c: set().union(*d.loc[d["conf"] == c, "names"]) for c in ("NeurIPS", "EMNLP")}
    both = sets["NeurIPS"] & sets["EMNLP"]
    e = d[d["conf"] == "EMNLP"]
    n = d[d["conf"] == "NeurIPS"]
    return {
        "authors_neurips": len(sets["NeurIPS"]), "authors_emnlp": len(sets["EMNLP"]), "authors_both": len(both),
        "share_emnlp_authors_also_neurips": round(len(both) / len(sets["EMNLP"]), 4),
        "share_neurips_authors_also_emnlp": round(len(both) / len(sets["NeurIPS"]), 4),
        "share_emnlp_papers_with_neurips_author": round(float(e["names"].map(lambda s: bool(s & both)).mean()), 4),
        "share_neurips_papers_with_emnlp_author": round(float(n["names"].map(lambda s: bool(s & both)).mean()), 4),
    }
out = {str(y): lens(df[df["year"] == y]) for y in (2025, 2026)}
n25 = df[df["year"] == 2025].groupby("conf").size()
d26 = df[df["year"] == 2026]
rng = np.random.default_rng(11)
draws = []
for _ in range(50):
    parts = [d26[d26["conf"] == c].sample(int(n25[c]), random_state=int(rng.integers(1e9))) for c in ("NeurIPS", "EMNLP")]
    draws.append(lens(pd.concat(parts)))
dr = pd.DataFrame(draws)
out["2026_at_2025_size"] = {c: {"mean": round(float(dr[c].mean()), 4), "lo": round(float(dr[c].quantile(.025)), 4),
                                "hi": round(float(dr[c].quantile(.975)), 4)} for c in dr if c.startswith("share")}
json.dump(out, open(OUT / "authors.json", "w"), indent=1)
print(json.dumps(out, indent=1))
