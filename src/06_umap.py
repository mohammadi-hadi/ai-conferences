"""2D layout of all four sets together (UMAP on the shared vectors, same settings as the atlases)."""
import numpy as np
import pandas as pd
import umap
from config import P, SEED

emb = np.load(P / "embeddings.npy")
xy = umap.UMAP(n_neighbors=30, min_dist=0.05, metric="cosine", random_state=SEED).fit_transform(emb)
pd.DataFrame(xy, columns=["x", "y"]).to_parquet(P / "umap.parquet")
print(xy.shape, xy.min(0), xy.max(0))
