"""Collect our runs and the published scores into one table.

    python harness/summarize.py      # from research/results/openproblems

Writes
  summary.csv  one row per (dataset, metric, method): source (ours /
               published), value, and the rank of each of our runs among the
               published non-control methods plus itself (1 = best)
  tables.md    per dataset: every metric x every method, with our runs first
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
CONTROLS = {"random_features", "spectral_features", "true_features"}
DATASETS = ["mouse_blood_olsson_labelled", "mouse_hspc_nestorowa2016", "pancreas", "zebrafish"]
METRICS = [  # as in the task's results page, grouped
    "qnx_auc", "qlocal", "qglobal", "qnx_at_k30", "lcmc_at_k30",
    "trustworthiness_at_k30", "continuity_at_k30",
    "waypoint_distance_correlation", "centroid_distance_correlation", "label_distance_correlation",
    "spectral_distance_correlation", "density_preservation",
    "normalized_mutual_information", "adjusted_rand_index",
]

pub = pd.read_csv(HERE / "published_scores.csv").assign(source="published")
ours = []
for ds in DATASETS:
    r = pd.read_csv(HERE / "runs" / f"{ds}.csv")
    ours.append(r.melt(id_vars=["dataset", "method"], var_name="metric", value_name="value"))
ours = pd.concat(ours).query("metric != 'time_s'").dropna(subset=["value"]).assign(source="ours")

# rank of each of our runs among the published non-control methods (+ itself)
ref = pub[~pub.method.isin(CONTROLS)]
def rank(row):
    if row.source != "ours":
        return pd.NA
    col = ref[(ref.dataset == row.dataset) & (ref.metric == row.metric)].value.dropna()
    return f"{int((col > row.value).sum()) + 1}/{len(col) + 1}"
allv = pd.concat([ours, pub])[["dataset", "metric", "method", "source", "value"]]
allv = allv[allv.metric.isin(METRICS)]
allv["rank"] = allv.apply(rank, axis=1)
allv.to_csv(HERE / "summary.csv", index=False)

def markdown(t):
    cells = t.map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    out = ["| metric | " + " | ".join(cells.columns) + " |", "|---|" + "---:|" * len(cells.columns)]
    out += [f"| {i} | " + " | ".join(r) + " |" for i, r in zip(cells.index, cells.values)]
    return "\n".join(out)


lines = ["# Open Problems: all scores", "",
         "Our runs are marked `(ours)`; the other columns are the published scores",
         "(run_2024-12-19_15-35-59). Controls are `random_features`, `spectral_features`",
         "and `true_features`. Empty cells: the metric was not computed or not published.", ""]
for ds in DATASETS:
    d = allv[allv.dataset == ds].copy()
    d["col"] = d.method + d.source.map({"ours": " (ours)", "published": ""})
    t = d.pivot_table(index="metric", columns="col", values="value", aggfunc="first").reindex(METRICS).dropna(how="all")
    o = sorted(c for c in t.columns if c.endswith("(ours)"))
    c = [x for x in t.columns if x not in o and x not in CONTROLS] + sorted(CONTROLS & set(t.columns))
    t = t[o + c]
    lines += [f"## {ds}", "", markdown(t), ""]
(HERE / "tables.md").write_text("\n".join(lines))
print(f"wrote {HERE / 'summary.csv'} and {HERE / 'tables.md'}")
