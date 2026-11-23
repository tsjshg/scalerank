"""Overall score of ScaleRank settings next to the published methods.

    python harness/sweep_summary.py runs/sweep/*.csv

Each metric is scaled per dataset to [0, 1] between the worst and the best
control (random_features, spectral_features, true_features), clipped, and the
scaled metrics are averaged, as in Open Problems' overall ranking; a metric a
method lacks counts as 0. spectral_distance_correlation is left out (not
computed on the large datasets).
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
CONTROLS = ["random_features", "spectral_features", "true_features"]
METRICS = [
    "qnx_auc", "qlocal", "qglobal", "qnx_at_k30", "lcmc_at_k30",
    "trustworthiness_at_k30", "continuity_at_k30",
    "waypoint_distance_correlation", "centroid_distance_correlation", "label_distance_correlation",
    "density_preservation", "normalized_mutual_information", "adjusted_rand_index",
]

pub = pd.read_csv(HERE / "published_scores.csv")
pub = pub[pub.metric.isin(METRICS)]
ours = pd.concat([pd.read_csv(f) for f in sys.argv[1:]])
ours = ours.melt(id_vars=["dataset", "method"], var_name="metric", value_name="value")
ours = ours[ours.metric.isin(METRICS)].assign(method=lambda d: "* " + d.method)

rows = []
for ds, o in ours.groupby("dataset"):
    p = pub[pub.dataset == ds]
    ctrl = p[p.method.isin(CONTROLS)].pivot_table(index="metric", columns="method", values="value")
    lo, hi = ctrl.min(axis=1), ctrl.max(axis=1)
    both = pd.concat([p[~p.method.isin(CONTROLS)], o])
    t = both.pivot_table(index="method", columns="metric", values="value").reindex(columns=METRICS)
    scaled = ((t - lo) / (hi - lo)).clip(0, 1).fillna(0)
    for m, v in scaled.mean(axis=1).items():
        rows.append({"dataset": ds, "method": m, "score": v,
                     "qnx_auc": t.loc[m, "qnx_auc"], "qglobal": t.loc[m, "qglobal"]})
r = pd.DataFrame(rows)
r["rank"] = r.groupby("dataset").score.rank(ascending=False, method="min").astype(int)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
w = r.pivot(index="method", columns="dataset", values="score")
w["mean"] = w.mean(axis=1)
print(w.sort_values("mean", ascending=False).round(3).to_string())
print()
print(r.pivot(index="method", columns="dataset", values="qnx_auc").round(3).sort_values(r.dataset.iloc[0], ascending=False).to_string())
