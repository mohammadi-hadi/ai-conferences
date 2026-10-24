"""Paths. The papers and their vectors come from the two atlas pipelines, which embed title + abstract with the
same open model (Qwen3-Embedding-0.6B, normalised, 384 tokens), so NeurIPS and EMNLP papers share one space."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT.parent
NEURIPS = DOCS / "NeurIPS2026_Analysis/data/processed"
EMNLP = DOCS / "EMNLP2026_Analysis/data/processed"
P = ROOT / "data/processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
for d in (P, OUT, FIG):
    d.mkdir(parents=True, exist_ok=True)

YEARS = (2025, 2026)
CONFS = ("NeurIPS", "EMNLP")
SEED = 7
