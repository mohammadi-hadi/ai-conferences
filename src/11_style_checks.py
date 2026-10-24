"""Is the convergence just a change in how abstracts are written?

Abstracts at both conferences changed style between 2025 and 2026 (far fewer "novel" and "enhance"), so two checks:
- titles only: a TF-IDF classifier on titles (no abstracts, no vectors), 3,000 papers per side, 20 draws;
- year direction removed: take out the average 2025 -> 2026 shift of all papers from every vector, then recompute the
  closest-paper shares (3,000 per side, 50 draws) and the common ground (40 topics, refit).
Also writes the common ground of NeurIPS 2026 with EMNLP 2025 and of NeurIPS 2025 with EMNLP 2026, for the
"who moved towards whom" paragraph.
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import P, OUT, SEED

N = 3000
df = pd.read_parquet(P / "papers.parquet")
emb = np.load(P / "embeddings.npy")
idx = {(c, y): np.flatnonzero(((df["conf"] == c) & (df["year"] == y)).to_numpy())
       for c in ("NeurIPS", "EMNLP") for y in (2025, 2026)}
q = lambda v: {"mean": round(float(np.mean(v)), 4), "lo": round(float(np.quantile(v, .025)), 4),
               "hi": round(float(np.quantile(v, .975)), 4)}
out = {}

# 1. titles only
for y in (2025, 2026):
    rng = np.random.default_rng(SEED + y)
    aucs = []
    for _ in range(20):
        a, b = rng.choice(idx[("NeurIPS", y)], N, replace=False), rng.choice(idx[("EMNLP", y)], N, replace=False)
        X = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit_transform(df["title"].iloc[np.r_[a, b]])
        yy = np.r_[np.zeros(N), np.ones(N)]
        cv = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1e9)))
        aucs.append(cross_val_score(LogisticRegression(C=4.0, max_iter=3000), X, yy, cv=cv, scoring="roc_auc").mean())
    out.setdefault("titles_only_auc", {})[str(y)] = q(aucs)
print("titles only", out["titles_only_auc"], flush=True)

# 2. year direction removed
yr = df["year"].to_numpy()
d = emb[yr == 2026].mean(0) - emb[yr == 2025].mean(0)
d /= np.linalg.norm(d)
E = emb - np.outer(emb @ d, d)
E /= np.linalg.norm(E, axis=1, keepdims=True)
for y in (2025, 2026):
    rng = np.random.default_rng(SEED + 100 + y)
    ne, en = [], []
    for _ in range(50):
        a, b = rng.choice(idx[("NeurIPS", y)], N, replace=False), rng.choice(idx[("EMNLP", y)], N, replace=False)
        X = E[np.r_[a, b]]
        S = X @ X.T
        np.fill_diagonal(S, -np.inf)
        top = S.argmax(1)
        ne.append((top[:N] >= N).mean())
        en.append((top[N:] < N).mean())
    out.setdefault("year_removed_closest", {})[str(y)] = {"neurips_closest_at_emnlp": q(ne), "emnlp_closest_at_neurips": q(en)}
lab = KMeans(40, n_init=10, random_state=SEED).fit_predict(E)
sh = df.assign(t=lab).groupby(["conf", "year"])["t"].value_counts(normalize=True).unstack(fill_value=0).T
out["year_removed_common_ground"] = {str(y): round(float(np.minimum(sh[("NeurIPS", y)], sh[("EMNLP", y)]).sum()), 4)
                                     for y in (2025, 2026)}
print("year removed", out["year_removed_closest"], out["year_removed_common_ground"], flush=True)

# 3. cross-year common ground on the main 40 topics
t = pd.read_parquet(P / "topics.parquet")["topic"]
s = df.assign(t=t).groupby(["conf", "year"])["t"].value_counts(normalize=True).unstack(fill_value=0).T
out["common_ground_cross_year"] = {f"NeurIPS {a} vs EMNLP {b}": round(float(np.minimum(s[("NeurIPS", a)], s[("EMNLP", b)]).sum()), 4)
                                   for a in (2025, 2026) for b in (2025, 2026)}
print(out["common_ground_cross_year"])
json.dump(out, open(OUT / "style_checks.json", "w"), indent=1)
