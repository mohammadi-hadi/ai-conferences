"""Does the headline survive other settings?

- topic overlap with 30 and 60 joint topics instead of 40;
- mixing index with 10 and 50 neighbours instead of 20 (20 draws each);
- the share of the topic-overlap gain that comes from agent and RL topics, for each topic count.
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from config import P, OUT, SEED

df = pd.read_parquet(P / "papers.parquet")
emb = np.load(P / "embeddings.npy")
out = {"topic_overlap": {}, "mixing_k": {}}
AGENT_WORDS = ("agent", "agentic", "reinforcement", "rlvr", "policy optimization", "reward")

for K in (30, 40, 60):
    lab = KMeans(K, n_init=10, random_state=SEED).fit_predict(emb)
    d = df.assign(t=lab)
    sh = d.groupby(["conf", "year"])["t"].value_counts(normalize=True).unstack(fill_value=0).T
    ov = {y: np.minimum(sh[("NeurIPS", y)], sh[("EMNLP", y)]) for y in (2025, 2026)}
    gain = ov[2026] - ov[2025]
    # a topic counts as agent/RL when at least 40% of its titles use one of those words
    hit = d["title"].str.lower().str.contains("|".join(AGENT_WORDS))
    agentish = d.assign(h=hit).groupby("t")["h"].mean() >= 0.4
    out["topic_overlap"][str(K)] = {
        "2025": round(float(ov[2025].sum()), 4), "2026": round(float(ov[2026].sum()), 4),
        "gain": round(float(gain.sum()), 4),
        "gain_from_agent_rl_topics": round(float(gain[agentish.reindex(gain.index, fill_value=False)].sum()), 4),
        "agent_rl_topics": int(agentish.sum())}
    print(K, out["topic_overlap"][str(K)], flush=True)

def mixing(year, k, rng, N=3000):
    a = rng.choice(np.flatnonzero(((df.conf == "NeurIPS") & (df.year == year)).to_numpy()), N, replace=False)
    b = rng.choice(np.flatnonzero(((df.conf == "EMNLP") & (df.year == year)).to_numpy()), N, replace=False)
    X = emb[np.r_[a, b]]
    y = np.r_[np.zeros(N), np.ones(N)]
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    return (y[nn] != y[:, None]).mean() / 0.5

for k in (10, 20, 50):
    out["mixing_k"][str(k)] = {}
    for year in (2025, 2026):
        rng = np.random.default_rng(SEED + year + k)
        v = np.array([mixing(year, k, rng) for _ in range(20)])
        out["mixing_k"][str(k)][str(year)] = {"mean": round(float(v.mean()), 4),
                                              "lo": round(float(np.quantile(v, .025)), 4),
                                              "hi": round(float(np.quantile(v, .975)), 4)}
    print(k, out["mixing_k"][str(k)], flush=True)

json.dump(out, open(OUT / "robustness.json", "w"), indent=1)
