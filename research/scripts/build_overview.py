"""Build the research notes (the illustrated overview), in English and Japanese.

Pulls the numbers the charts need from drbench's ``results/`` and injects
them into ``research/docs/overview_template_en.html`` (English) and
``research/docs/overview_template.html`` (Japanese), writing the published
pages ``web/public/docs/index.html`` and ``web/public/docs/ja.html`` of a
scalerank-web checkout next to this repository (``--out`` writes them
elsewhere).

Needed runs (``drbench run -m ...``, with this package installed): the
baselines, the current ScaleRank default at gamma 1/0.5/0, and the first
prototype's runs under their explicit names (see ALIASES).  Figures quoted
from one-off investigations (2022 notebook scores, force decomposition,
hubness, the Q sweep) are written in the template itself.

Further runs for the later sections: the density term (the DENS specs, seeds
0-2, and their seed stability in ``results/stability_density.csv`` from
``scripts/stability_density.py``); the tuned t-SNE / UMAP grid (see
``scripts/compare_tuned_baselines.py``), ``scalerank_fast`` and
``scalerank_rs`` / ``tsne_rs`` / ``umap_rs`` on the 8 datasets, and the 70k
runs on ``mnist70k`` / ``fmnist70k`` (seed 0); the stability section reads
``results/stability*.csv`` from drbench's ``scripts/stability.py``.

    python scripts/build_overview.py [--out DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from drbench import datasets
from drbench.metrics import local_radius
from drbench.runner import RESULTS_DIR, _paths, load_results, summarize

ROOT = Path(__file__).resolve().parents[1]  # research/
PUBLISHED = ROOT.parent.parent / "scalerank-web" / "web" / "public" / "docs"  # the app serves them
BASE = ["pca", "mds", "tsne", "umap", "densmap", "pacmap", "trimap"]
FIRST = "hard=0.3,n_neg=1,per_point=32"  # the first prototype: triplets + hard negatives
# Short keys used by the page -> drbench method specs.
ALIASES = {
    "sr:1.0": "scalerank:gamma=1.0",  # current default (softmax, n_neg=20, per_point=8, hard=0)
    "sr:0.5": "scalerank:gamma=0.5",
    "sr:0.0": "scalerank:gamma=0.0",
    "v1:1.0": f"scalerank:gamma=1.0,{FIRST}",
    "v1:0.5": f"scalerank:gamma=0.5,{FIRST}",
    "v1:0.0": f"scalerank:gamma=0.0,{FIRST}",
    "v1:nohard": "scalerank:gamma=1.0,hard=0.0,n_neg=1,per_point=32",
    "v1:R": "scalerank:gamma=1.0,local_scaling=10,local_scaling_kmax=30,hard=0.15,n_neg=1,per_point=32",
    "q100": "scalerank:gamma=1.0,hard=0.0,n_neg=100,per_point=4",
}
ORDER = ["s_curve", "hier_blobs", "density_blobs", "digits", "pollen", "usoskin", "kolodziejczyk", "buettner"]
METRICS = ["rnx@10", "auc_rnx", "trust@10", "dist_spearman", "density_corr", "knn_acc", "time_s"]
DIFF_METRICS = ["rnx@10", "auc_rnx", "dist_spearman", "density_corr", "knn_acc"]
TUNED = (
    [f"tsne:perplexity={p},exaggeration={e}" for p in (5, 30, 100, 300) for e in (1, 2, 4)]
    + ["tsne:initialization=random"]
    + [f"umap:n_neighbors={k},min_dist={md}" for k in (5, 15, 50, 200) for md in (0.0, 0.1, 0.5, 0.99)]
    + ["umap:init=random"]
)
# 70k runs: page key -> drbench spec
BIG = {
    "tsne": "tsne",
    "umap": "umap",
    "rs:1.0": "scalerank_rs:gamma=1.0",
    "rs:1.0ns": "scalerank_rs:gamma=1.0,neg_scale=0.5",
    "rs:0.0": "scalerank_rs:gamma=0.0",
}
BIG_DATA = ["mnist70k", "fmnist70k"]
# (Rust or fast version, reference) pairs compared over the 8 datasets
PORTS = [
    ("tsne_rs", "tsne"),
    ("umap_rs", "umap:init=pca"),
    ("scalerank_rs:gamma=1.0", "scalerank_fast:gamma=1.0"),
]
FAST = [(f"scalerank_fast:gamma={g}", f"scalerank:gamma={g}") for g in ("1.0", "0.5", "0.0")]


def spec(key):
    return ALIASES.get(key, key)


def unit_square(Y):
    Y = Y - Y.min(0)
    return np.round(Y / Y.max(), 3).tolist()


def outlier_mask(Y):
    r = local_radius(Y, 10)
    return r / np.median(r) > 3


def k_occurrence(X, k=10):
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    return np.bincount(idx.ravel(), minlength=len(X))


def emb(ds, key, seed=0):
    return np.load(_paths(ds, spec(key), seed)["emb"])


def mean_diff(table, a, b):
    """Mean over the 8 datasets of (a - b) per metric, and the mean run times."""
    ta = table[(table.method == a) & table.dataset.isin(ORDER)].set_index("dataset")
    tb = table[(table.method == b) & table.dataset.isin(ORDER)].set_index("dataset")
    common = ta.index.intersection(tb.index)
    d = (ta.loc[common, DIFF_METRICS] - tb.loc[common, DIFF_METRICS]).mean()
    return {
        "n": len(common),
        "diff": {k: (None if v != v else round(float(v), 3)) for k, v in d.items()},
        "time": [round(float(ta.loc[common, "time_s"].mean()), 2), round(float(tb.loc[common, "time_s"].mean()), 2)],
    }


def big_data(table):
    out = {"summary": {}, "emb": {}}
    for ds in BIG_DATA:
        out["summary"][ds] = {}
        for key, sp in BIG.items():
            r = table[(table.dataset == ds) & (table.method == sp)]
            if not r.empty:
                r = r.iloc[0]
                out["summary"][ds][key] = {k: round(float(r[k]), 3) for k in METRICS if k in r and r[k] == r[k]}
    # embeddings of MNIST, 5000 random points (the same for every method)
    d = datasets.load("mnist70k")
    sub = np.sort(np.random.default_rng(0).choice(d.n, 5000, replace=False))
    out["emb"] = {
        "labels": d.labels[sub].astype(int).tolist(),
        "emb": {k: unit_square(np.load(_paths("mnist70k", sp, 0)["emb"])[sub]) for k, sp in BIG.items() if k != "rs:1.0"},
    }
    return out


def stability_data():
    """Seed and row-order stability per method (results/stability*.csv from scripts/stability.py)."""
    path = RESULTS_DIR / "stability.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    out = {}
    for (exp, method), g in df.groupby(["experiment", "method"]):
        out.setdefault(method, {})[exp] = {k: round(float(g[k].mean()), 3) for k in ("procrustes", "knn", "dist")}
    pts = RESULTS_DIR / "stability_points.csv"
    rho = pd.read_csv(pts).groupby("method")["rho"].mean().round(2).to_dict() if pts.exists() else {}
    return {"methods": out, "rho": rho}


OP_DIR = ROOT / "results" / "openproblems"
OP_DATA = ["mouse_blood_olsson_labelled", "mouse_hspc_nestorowa2016", "pancreas", "zebrafish"]
OP_CONTROLS = ["random_features", "spectral_features", "true_features"]
OP_METRICS = [
    "qnx_auc", "qlocal", "qglobal", "qnx_at_k30", "lcmc_at_k30",
    "trustworthiness_at_k30", "continuity_at_k30",
    "waypoint_distance_correlation", "centroid_distance_correlation", "label_distance_correlation",
    "density_preservation", "normalized_mutual_information", "adjusted_rand_index",
]
_SR = "gamma=0.5,n_hvg=3000,n_pcs=100"
# page key -> Open Problems harness spec (sorted key=value items, without seed)
OP_SETTINGS = {
    "sr_dens": f"sr:dens_all=1,dens_lambda=4,dens_pcs=300,{_SR}",
    "sr_dens100": f"sr:dens_all=1,dens_lambda=4,{_SR}",
    "sr_dens_own": f"sr:dens_all=1,dens_lambda=4,dens_own=1,dens_pcs=300,{_SR}",
    "sr": f"sr:{_SR}",
    "densmap": "task:densmap",
    "tsne": "task:tsne",
    "phate": "task:phate",
    "umap": "umap:",  # the task's default, run as `umap` in runs/<dataset>.csv
    "pca": "task:pca",
}


def _op_setting(method):
    """'sr:a=1,seed=3' -> ('sr:a=1', 3), with the items sorted; missing seed = 42."""
    head, _, rest = method.partition(":")
    parts = rest.split(",")
    items = dict(p.split("=") for p in parts if "=" in p)
    seed = int(items.pop("seed", 42))
    base = [p for p in parts if "=" not in p]
    return f"{head}:" + ",".join(base + [f"{k}={v}" for k, v in sorted(items.items())]), seed


def openproblems_data():
    """Overall Open Problems score (scaled between the controls, as on the
    task's leaderboard) and density_preservation per setting, mean and sd over
    seeds, from research/results/openproblems/runs."""
    files = sorted((OP_DIR / "runs").glob("*.csv"))
    files += [f for sub in ("hvg", "sweep", "dens", "seeds") for f in sorted((OP_DIR / "runs" / sub).glob("*.csv"))]
    runs = pd.concat([pd.read_csv(f) for f in files])
    runs[["setting", "seed"]] = runs.method.apply(lambda m: pd.Series(_op_setting(m)))
    key = {v: k for k, v in OP_SETTINGS.items()}
    runs = runs[runs.setting.isin(key)].drop_duplicates(["dataset", "setting", "seed"], keep="last")
    pub = pd.read_csv(OP_DIR / "published_scores.csv")
    rows = []
    for ds, r in runs.groupby("dataset"):
        ctrl = pub[(pub.dataset == ds) & pub.method.isin(OP_CONTROLS)].pivot_table(index="metric", columns="method", values="value")
        lo, hi = ctrl.min(axis=1)[OP_METRICS], ctrl.max(axis=1)[OP_METRICS]
        scaled = ((r[OP_METRICS] - lo) / (hi - lo)).clip(0, 1).fillna(0)
        rows.append(pd.DataFrame({"dataset": ds, "key": r.setting.map(key).values, "seed": r.seed.values,
                                  "score": scaled.mean(axis=1).values, "density": r.density_preservation.values}))
    s = pd.concat(rows)
    out = {}
    for k in OP_SETTINGS:
        g = s[s.key == k]
        per = g.pivot_table(index="seed", columns="dataset", values="score").reindex(columns=OP_DATA)
        full = per.dropna()
        dens = g.groupby("dataset").density.mean().reindex(OP_DATA)
        out[k] = {
            "score": {ds: round(float(per[ds].mean()), 3) for ds in OP_DATA},
            "mean": round(float(full.mean(axis=1).mean()), 3),
            "sd": round(float(full.mean(axis=1).std()), 3) if len(full) > 1 else None,
            "seeds": len(full),
            "density": {ds: round(float(v), 3) for ds, v in dens.items()},
        }
    return out


# drbench: the density term against the same settings without it (8 datasets, seeds 0-2)
DENS = {
    "rs:1.0": "scalerank_rs:gamma=1.0",
    "rs:1.0d1": "scalerank_rs:gamma=1.0,dens_lambda=1",
    "rs:1.0d4": "scalerank_rs:gamma=1.0,dens_lambda=4",
    "rs:1.0o1": "scalerank_rs:gamma=1.0,dens_lambda=1,dens_own=True",
    "rs:1.0o4": "scalerank_rs:gamma=1.0,dens_lambda=4,dens_own=True",
    "rs:0.5": "scalerank_rs:gamma=0.5",
    "rs:0.5d4": "scalerank_rs:gamma=0.5,dens_lambda=4",
    "rs:0.5o4": "scalerank_rs:gamma=0.5,dens_lambda=4,dens_own=True",
    "densmap": "densmap",
    "tsne": "tsne",
    "umap": "umap",
}


def density_data(table):
    """Mean over the 8 datasets of each metric per setting (settings with all 8 only),
    with its seed stability (results/stability*.csv) when measured."""
    st = pd.concat([pd.read_csv(f) for f in (RESULTS_DIR / "stability.csv", RESULTS_DIR / "stability_density.csv") if f.exists()])
    st = st[st.experiment == "seed"].groupby("method")[["knn", "dist"]].mean()
    out = {}
    for key, sp in DENS.items():
        t = table[(table.method == sp) & table.dataset.isin(ORDER)]
        if t.dataset.nunique() < len(ORDER):
            continue
        out[key] = {m: round(float(t[m].mean()), 3) for m in METRICS}
        if sp in st.index:
            out[key]["stab_knn"] = round(float(st.loc[sp, "knn"]), 3)
            out[key]["stab_dist"] = round(float(st.loc[sp, "dist"]), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PUBLISHED, help="where to write index.html and ja.html (default: ../scalerank-web/web/public/docs)")
    args = ap.parse_args()
    table = summarize(load_results()).reset_index()
    summary = {}
    for ds in ORDER:
        summary[ds] = {}
        for key in BASE + list(ALIASES) + ["random"]:
            r = table[(table.dataset == ds) & (table.method == spec(key))]
            if r.empty:
                continue
            r = r.iloc[0]
            summary[ds][key] = {k: (None if k not in r or r[k] != r[k] else round(float(r[k]), 3)) for k in METRICS}

    curves = {}
    for ds in ["pollen", "digits"]:
        n = datasets.load(ds).n
        grid = np.unique(np.round(np.geomspace(1, n - 2, 70)).astype(int))
        curves[ds] = {}
        for key in BASE + ["sr:1.0", "sr:0.0", "v1:1.0"]:
            r = np.load(_paths(ds, spec(key), 0)["curves"])["r_nx"]
            curves[ds][key] = {"k": grid.tolist(), "r": [round(float(r[k]), 4) for k in grid]}

    digits = datasets.load("digits")
    hier = datasets.load("hier_blobs")
    hier_keys = ["v1:1.0", "v1:nohard", "sr:1.0", "tsne"]
    embeddings = {
        "digits": {
            "labels": digits.labels.tolist(),
            "emb": {k: unit_square(emb("digits", k)) for k in ["mds", "tsne", "v1:1.0", "sr:1.0"]},
        },
        "hier_blobs": {
            "labels": hier.labels_coarse.tolist(),
            "antihub": (k_occurrence(hier.X) <= 2).astype(int).tolist(),
            "emb": {k: unit_square(emb("hier_blobs", k)) for k in hier_keys},
            "outlier": {k: outlier_mask(emb("hier_blobs", k)).astype(int).tolist() for k in hier_keys},
        },
    }

    outfrac = {}
    for ds in ORDER:
        outfrac[ds] = {}
        for key in ["v1:1.0", "v1:0.5", "v1:0.0", "v1:R", "sr:1.0", "sr:0.5", "sr:0.0", "q100", "tsne", "mds"]:
            fr = [outlier_mask(np.load(p)).mean() for s in (0, 1, 2) if (p := _paths(ds, spec(key), s)["emb"]).exists()]
            outfrac[ds][key] = round(float(np.mean(fr)), 4) if fr else None

    tuned = {}
    for ds in ORDER:
        rows = table[(table.dataset == ds) & table.method.isin(TUNED)]
        tuned[ds] = [[r.method, round(float(r.dist_spearman), 3), round(float(r["rnx@10"]), 3)] for _, r in rows.iterrows()]

    data = {
        "summary": summary,
        "curves": curves,
        "emb": embeddings,
        "outfrac": outfrac,
        "tuned": tuned,
        "fast": {a.split("=")[1]: mean_diff(table, a, b) for a, b in FAST},
        "ports": [{"rust": a, "ref": b, **mean_diff(table, a, b)} for a, b in PORTS],
        "big": big_data(table),
        "stability": stability_data(),
        "op": openproblems_data(),
        "dens": density_data(table),
    }
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    args.out.mkdir(parents=True, exist_ok=True)
    for template, out in [("overview_template_en.html", "index.html"), ("overview_template.html", "ja.html")]:
        src = ROOT / "docs" / template
        dst = args.out / out
        lang = "en" if template.endswith("_en.html") else "ja"
        # the templates are page fragments (no doctype, <html> or charset); make them standalone documents
        head = f'<!doctype html>\n<html lang="{lang}">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        dst.write_text(head + src.read_text().replace("/*__DATA__*/", payload) + "\n</html>\n")
        print(f"written {dst} ({dst.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
