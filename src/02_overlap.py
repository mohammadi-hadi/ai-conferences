"""Did NeurIPS and EMNLP papers move closer from 2025 to 2026?

Each draw takes the same number of papers from each conference and year (3,000, without replacement), so a
bigger conference or a bigger year cannot change the numbers by itself. Fifty draws give the spread.

- mixing: for every paper, the share of its 20 nearest papers that come from the other conference, averaged
  and divided by 0.5, so 0 means the two never touch and 1 means they are fully mixed;
- auc: how well a linear classifier tells the two conferences apart from the vector alone (5-fold), so 1 means
  always and 0.5 means a coin flip;
- nn_sim: the median, over papers, of the cosine similarity to the closest paper of the other conference;
- centroid: cosine similarity of the two conferences' average vectors.
nn_sim and centroid are absolute and rise whenever every set gets more uniform; mixing and auc are the headline ones.
The 95% ranges show the spread over draws, not the full uncertainty (EMNLP 2025 draws 3,000 of its 3,484 papers).
"""
import json, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import P, OUT, SEED, YEARS

N, K, DRAWS, AUC_DRAWS = 3000, 20, 50, 20
df = pd.read_parquet(P / "papers.parquet")
emb = np.load(P / "embeddings.npy")
mask = np.ones(len(df), bool)
tag = sys.argv[1] if len(sys.argv) > 1 else "all"
if tag == "main":  # robustness: EMNLP main track only
    mask = ~((df["conf"] == "EMNLP") & (df["track"] != "Main")).to_numpy()

sizes = df[mask].groupby(["conf", "year"]).size()
N = min(N, int(sizes.min()))  # the smallest group sets the draw size (EMNLP 2025 main track has 1,809)

def draw(year, rng):
    a = np.flatnonzero(mask & (df["conf"] == "NeurIPS").to_numpy() & (df["year"] == year).to_numpy())
    b = np.flatnonzero(mask & (df["conf"] == "EMNLP").to_numpy() & (df["year"] == year).to_numpy())
    return rng.choice(a, N, replace=False), rng.choice(b, N, replace=False)

def measures(ia, ib, with_auc, rng):
    X = emb[np.concatenate([ia, ib])]
    y = np.r_[np.zeros(N), np.ones(N)]
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, K, axis=1)[:, :K]
    other = (y[nn] != y[:, None]).mean(axis=1)
    cross = S[:N, N:]
    nn_sim = np.median(np.r_[cross.max(axis=1), cross.max(axis=0)])
    ca, cb = X[:N].mean(0), X[N:].mean(0)
    out = {"mixing": float(other.mean() / 0.5), "nn_sim": float(nn_sim),
           "centroid": float(ca @ cb / np.linalg.norm(ca) / np.linalg.norm(cb))}
    if with_auc:
        cv = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1e9)))
        out["auc"] = float(cross_val_score(LogisticRegression(C=1.0, max_iter=2000), X, y, cv=cv,
                                           scoring="roc_auc").mean())
    return out

rows = []
for year in YEARS:
    rng = np.random.default_rng(SEED + year)
    for d in range(DRAWS):
        ia, ib = draw(year, rng)
        rows.append({"year": year, "draw": d, **measures(ia, ib, d < AUC_DRAWS, rng)})
        print(year, d, {k: round(v, 4) for k, v in rows[-1].items() if k not in ("year", "draw")}, flush=True)

r = pd.DataFrame(rows)
r.to_csv(OUT / f"overlap_draws_{tag}.csv", index=False)
summ = {}
for m in ("mixing", "auc", "nn_sim", "centroid"):
    summ[m] = {}
    for y in YEARS:
        v = r.loc[r["year"] == y, m].dropna()
        summ[m][str(y)] = {"mean": round(float(v.mean()), 4), "lo": round(float(v.quantile(.025)), 4),
                           "hi": round(float(v.quantile(.975)), 4), "n_draws": int(len(v))}
json.dump({"tag": tag, "papers_per_conf_per_draw": N, "k": K, "measures": summ},
          open(OUT / f"overlap_{tag}.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
