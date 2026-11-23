"""Overall score per setting over several seeds.

    python harness/seed_summary.py runs/seeds/*.csv runs/dens/*.csv runs/sweep/*.csv runs/hvg/*.csv

Specs are grouped into settings by their key=value items without `seed`
(missing seed = 42), so the same setting from different files counts once per
seed (the last file wins). Scores are scaled as in sweep_summary.py. Prints, per
setting, the mean over seeds of each dataset's score and of the four-dataset
mean, with the standard deviation over seeds and the number of seeds.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent.parent
CONTROLS = ["random_features", "spectral_features", "true_features"]
METRICS = [
    "qnx_auc", "qlocal", "qglobal", "qnx_at_k30", "lcmc_at_k30",
    "trustworthiness_at_k30", "continuity_at_k30",
    "waypoint_distance_correlation", "centroid_distance_correlation", "label_distance_correlation",
    "density_preservation", "normalized_mutual_information", "adjusted_rand_index",
]
SETTINGS = {  # key=value items (without seed) -> label
    "sr:gamma=0.5,n_hvg=3000,n_pcs=100": "ScaleRank, no density term",
    "sr:dens_all=1,dens_lambda=4,gamma=0.5,n_hvg=3000,n_pcs=100": "density: PCA-100 target, input's neighbours",
    "sr:dens_all=1,dens_lambda=4,dens_pcs=300,gamma=0.5,n_hvg=3000,n_pcs=100": "density: PCA-300 target, input's neighbours",
    "sr:dens_all=1,dens_lambda=4,dens_own=1,dens_pcs=300,gamma=0.5,n_hvg=3000,n_pcs=100": "density: PCA-300 target, own neighbours",
    "task:densmap": "densMAP (task default)",
}


def split(spec):
    """'sr:a=1,seed=3' -> ('sr:a=1', 3); 'task:densmap,seed=2' -> ('task:densmap', 2)."""
    head, _, rest = spec.partition(":")
    parts = rest.split(",")
    base = [p for p in parts if "=" not in p]
    items = dict(p.split("=") for p in parts if "=" in p)
    seed = int(items.pop("seed", 42))
    return f"{head}:" + ",".join(base + [f"{k}={v}" for k, v in sorted(items.items())]), seed


runs = pd.concat([pd.read_csv(f) for f in sys.argv[1:]])
runs[["setting", "seed"]] = runs.method.apply(lambda m: pd.Series(split(m)))
runs = runs[runs.setting.isin(SETTINGS)].drop_duplicates(["dataset", "setting", "seed"], keep="last")
pub = pd.read_csv(HERE / "published_scores.csv")
rows = []
for ds, r in runs.groupby("dataset"):
    p = pub[pub.dataset == ds]
    ctrl = p[p.method.isin(CONTROLS)].pivot_table(index="metric", columns="method", values="value")
    lo, hi = ctrl.min(axis=1)[METRICS], ctrl.max(axis=1)[METRICS]
    scaled = ((r[METRICS] - lo) / (hi - lo)).clip(0, 1).fillna(0)
    rows.append(r[["dataset", "setting", "seed"]].assign(score=scaled.mean(axis=1).values,
                                                       density=r.density_preservation.values))
s = pd.concat(rows)
s["setting"] = s.setting.map(SETTINGS)
per = s.pivot_table(index=["setting", "seed"], columns="dataset", values="score")
full = per.dropna()
full = full.assign(mean=full.mean(axis=1))
agg = full.groupby("setting").agg(["mean", "std"])
out = pd.DataFrame({c: agg[(c, "mean")].map("{:.3f}".format) + " ± " + agg[(c, "std")].map("{:.3f}".format)
                    for c in full.columns})
out["seeds"] = full.groupby("setting").size()
pd.set_option("display.width", 250)
print(out.sort_values("mean", ascending=False).to_string())
print()
d = s.pivot_table(index=["setting", "seed"], columns="dataset", values="density").dropna().groupby("setting").agg(["mean", "std"])
print("density_preservation, mean ± sd over seeds")
print(pd.DataFrame({c: d[(c, "mean")].map("{:.3f}".format) + " ± " + d[(c, "std")].map("{:.3f}".format)
                    for c in full.columns if c != "mean"}).to_string())
