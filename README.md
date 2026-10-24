# AI Conferences: NeurIPS and EMNLP on one map

Did NeurIPS and EMNLP move closer from 2025 to 2026? This repository puts all 23,984 papers of both conferences
in both years on one map and measures it.

**Read the story and open the map:** https://mohammadi.cv/ai-conferences/2026/

## Main results

- In 2025, 23% of NeurIPS papers had their closest paper at EMNLP. In 2026 it is 29%. For EMNLP papers with their
  closest paper at NeurIPS, it went from 22% to 25%. The closest-paper and classifier comparisons use equal-size
  samples (3,000 papers per conference, 50 draws; 20 for the classifier).
- The common ground across 40 shared topics went from 40% to 45%. Agent and RL topics gave the biggest gains.
- Most of the gain is both conferences moving to the same new topics; NeurIPS also moved a little towards EMNLP's 2025 topics.
- Cutting the joint map into small patches, 25% of papers sat in patches both conferences share about equally in
  2025, and 32% in 2026 (31% at 2025 sizes).
- At equal conference sizes, the share of authors with a paper at both did not go up.

## Data

The papers and their vectors come from the two atlas pipelines, which embed title and abstract with the same open
model (Qwen3-Embedding-0.6B):

- NeurIPS: https://github.com/mohammadi-hadi/neurips (main, datasets and evaluations, position and journal tracks)
- EMNLP: https://github.com/mohammadi-hadi/emnlp (main, Findings, industry and demo tracks, from the ACL Anthology)

| | 2025 | 2026 |
|---|---:|---:|
| NeurIPS | 5,856 | 9,110 |
| EMNLP | 3,484 | 5,534 |

## Steps

Clone the two atlas repos next to this one as `NeurIPS2026_Analysis/` and `EMNLP2026_Analysis/`
(`git clone https://github.com/mohammadi-hadi/neurips NeurIPS2026_Analysis`, the same for `emnlp`), run their
pipelines, then run the scripts below from `src/` with the same Python environment. `01_load.py` writes this repo's
`data/processed/papers.parquet` and `embeddings.npy`. The NeurIPS lists do not split tracks, so NeurIPS papers have
track `All`.

| Script | What it does |
|---|---|
| `01_load.py` | one table and one set of vectors for all four sets |
| `02_overlap.py` | mixing index, classifier AUC, nearest-paper similarity, centroid similarity per year (`all` or `main`) |
| `03_direction.py` | every pairing of years, to see who moved towards whom |
| `04_topics.py` | 40 joint topics, shares per conference and year, common ground |
| `05_authors.py` | author overlap, also at equal conference sizes |
| `06_umap.py` | 2D layout of all papers |
| `07_robustness.py` | 30 and 60 topics, 10 and 50 neighbours, EMNLP main track only |
| `11_style_checks.py` | titles only, the 2025 to 2026 shift taken out, common ground across years |
| `12_overlap_areas.py` | where on the map the two overlap, and each paper's neighbourhood mix (run before `09_map.py`) |
| `08_figures.py` | story figures |
| `09_map.py` | interactive map |
| `10_build_site.py` | story page and hub page in `docs/` |

Topic names are in `data/labels/topic_names.json`. Every number in the story is in `outputs/`.

## Licence

Code: MIT. Paper metadata: NeurIPS paper lists; ACL Anthology (CC BY 4.0).

Hadi Mohammadi · https://mohammadi.cv
