"""Who moved towards whom, and one plain number for the story.

Same balanced draws as 02 (3,000 papers per side, 50 draws). For every pairing of a NeurIPS year with an EMNLP year,
it reports the mixing index and the share of each conference's papers whose single closest paper is at the other one.
If NeurIPS 2026 vs EMNLP 2025 mixes more than NeurIPS 2025 vs EMNLP 2025, NeurIPS moved towards EMNLP's 2025 topics,
and the other way round.
"""
import json
import numpy as np
import pandas as pd
from config import P, OUT, SEED

N, K, DRAWS = 3000, 20, 50
df = pd.read_parquet(P / "papers.parquet")
emb = np.load(P / "embeddings.npy")
idx = {(c, y): np.flatnonzero(((df["conf"] == c) & (df["year"] == y)).to_numpy()) for c in ("NeurIPS", "EMNLP")
       for y in (2025, 2026)}

def one(ya, yb, rng):
    ia = rng.choice(idx[("NeurIPS", ya)], N, replace=False)
    ib = rng.choice(idx[("EMNLP", yb)], N, replace=False)
    X = emb[np.concatenate([ia, ib])]
    y = np.r_[np.zeros(N), np.ones(N)]
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, K, axis=1)[:, :K]
    other = (y[nn] != y[:, None]).mean(axis=1)
    top = S.argmax(axis=1)
    cross_top = y[top] != y
    return {"mixing": other.mean() / 0.5,
            "neurips_closest_at_emnlp": cross_top[:N].mean(), "emnlp_closest_at_neurips": cross_top[N:].mean()}

res = {}
for ya in (2025, 2026):
    for yb in (2025, 2026):
        rng = np.random.default_rng(SEED + 10 * ya + yb)
        r = pd.DataFrame([one(ya, yb, rng) for _ in range(DRAWS)])
        res[f"NeurIPS {ya} vs EMNLP {yb}"] = {c: {"mean": round(float(r[c].mean()), 4),
                                                   "lo": round(float(r[c].quantile(.025)), 4),
                                                   "hi": round(float(r[c].quantile(.975)), 4)} for c in r}
json.dump({"papers_per_side": N, "k": K, "draws": DRAWS, "pairs": res}, open(OUT / "direction.json", "w"), indent=1)
for k, v in res.items():
    print(k, {c: (d["mean"], d["lo"], d["hi"]) for c, d in v.items()})
