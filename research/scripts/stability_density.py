"""Seed stability of ScaleRank with the density term, for the research notes.

Uses drbench's scripts/stability.py (the seed experiment only) and writes
results/stability_density.csv, leaving results/stability.csv alone. Run from
the drbench folder after `drbench run` of the specs (seeds 0-2):

    uv run python path/to/scalerank/research/scripts/stability_density.py SPEC [SPEC ...]
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")  # drbench's scripts/
import stability as st  # noqa: E402

from drbench.runner import RESULTS_DIR  # noqa: E402

rows = st.seed_rows(np.random.default_rng(0), sys.argv[1:])
df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "stability_density.csv", index=False)
print(df.groupby("method", sort=False)[["procrustes", "knn", "dist"]].mean().round(3).to_string())
