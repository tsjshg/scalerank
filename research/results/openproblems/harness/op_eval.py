"""Score embeddings on Open Problems' dimensionality-reduction task, locally.

Mirrors task_dimensionality_reduction (log_cp10k datasets): the methods'
input handling (layers['normalized'], top-1000 HVG, optional PCA-50 via
scanpy), the embedding processor (waypoint and centroid distances), and the
metrics. The R co-ranking metric is re-implemented from coRanking's
definitions (Q_NX, LCMC, kmax = argmax LCMC, T&C at K=30) on the waypoint
distance matrices; the others follow the task's Python scripts.

    python op_eval.py DATA_DIR OUT_CSV [method ...]
"""

import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scipy.spatial
import scipy.stats
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_distances


# ---------------------------------------------------------------- methods
def hvg_matrix(adata, n_hvg=1000):
    X = adata.layers["normalized"]
    idx = adata.var["hvg_score"].to_numpy().argsort()[::-1][:n_hvg]
    return X[:, idx]


def pca50(X):
    return sc.tl.pca(X, n_comps=50, svd_solver="arpack")


def dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def m_umap(adata):  # task default: 1000 HVG, no PCA, random_state=42
    from umap import UMAP

    return UMAP(densmap=False, random_state=42).fit_transform(hvg_matrix(adata))


def m_tsne(adata):  # 1000 HVG, PCA-50 (NOT the task default, which uses all genes: see task:tsne)
    a = ad.AnnData(obs=adata.obs[[]])
    a.obsm["X_pca"] = pca50(hvg_matrix(adata))
    sc.tl.tsne(a, use_rep="X_pca", n_pcs=50)
    return a.obsm["X_tsne"]


# ---- the task's baselines with their argument defaults (src/methods/*/config.vsh.yaml);
# spec 'task:<method>' or 'task:<method>,n_hvg=3000,...'; n_hvg=0: all genes, n_pcs=0: no PCA
TASK_DEFAULTS = {
    "umap": {"n_hvg": 1000, "n_pcs": 0},
    "densmap": {"n_hvg": 0, "n_pcs": 0},
    "tsne": {"n_hvg": 0, "n_pcs": 50},
    "pca": {"n_hvg": 0},
    "phate": {"n_hvg": 0, "n_pcs": 50, "gamma": 1.0},
}


def task_method(name, n_hvg, n_pcs=0, gamma=1.0, seed=42):
    def run(adata):
        X = hvg_matrix(adata, n_hvg) if n_hvg else adata.layers["normalized"]
        if name in ("umap", "densmap"):
            from umap import UMAP

            Z = sc.tl.pca(X, n_comps=n_pcs, svd_solver="arpack") if n_pcs else X
            return UMAP(densmap=name == "densmap", random_state=seed).fit_transform(Z)
        if name == "tsne":
            a = ad.AnnData(obs=adata.obs[[]])
            a.obsm["X_pca"] = sc.tl.pca(X, n_comps=n_pcs, svd_solver="arpack")
            sc.tl.tsne(a, use_rep="X_pca", n_pcs=n_pcs)
            return a.obsm["X_tsne"]
        if name == "pca":
            return sc.pp.pca(X, n_comps=2, svd_solver="arpack")[:, :2]
        if name == "phate":
            from phate import PHATE

            return PHATE(n_pca=n_pcs, verbose=False, n_jobs=-1, gamma=gamma).fit_transform(X)
        raise KeyError(name)

    return run


def log_radii(Z, k=30):
    """ln(mean squared distance to the k nearest neighbours), ScaleRank's density target."""
    from sklearn.neighbors import NearestNeighbors

    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    return np.log(np.maximum((np.asarray(d[:, 1:], dtype=np.float64) ** 2).mean(axis=1), 1e-300))


_CACHE = {}  # PCA inputs and density targets, shared by specs within a run


def _cached(key, f):
    if key not in _CACHE:
        _CACHE[key] = f()
    return _CACHE[key]


def scalerank(n_hvg=1000, n_pcs=50, dens_all=0, dens_pcs=-1, seed=42, **kw):
    """ScaleRank on PCA-n_pcs of the top-n_hvg genes (n_hvg=0: all genes).

    dens_all=1: the density term's target radii come from all genes, via PCA-dens_pcs
    (-1: n_pcs; 0: no PCA, exact neighbours on the sparse matrix); the ranks still
    come from the HVG input."""
    def run(adata):
        import scalerank_rs

        pc = lambda X, k=n_pcs: np.ascontiguousarray(sc.tl.pca(X, n_comps=k, svd_solver="arpack"), dtype=np.float64)
        Z = _cached((id(adata), "Z", n_hvg, n_pcs), lambda: pc(hvg_matrix(adata, n_hvg) if n_hvg else adata.layers["normalized"]))
        target = None
        if dens_all:
            X, k = adata.layers["normalized"], int(kw.get("dens_k", 30))
            kp = n_pcs if dens_pcs < 0 else dens_pcs
            target = _cached((id(adata), "t", kp, k), lambda: log_radii(sp.csr_matrix(X) if kp == 0 else pc(X, kp), k))
        return scalerank_rs.ScaleRankFast(random_state=seed, **kw).fit_transform(Z, dens_target=target)

    return run


def method(name):
    """A name from METHODS, or a spec 'sr:key=val,...' for a ScaleRank variant,
    e.g. 'sr:n_hvg=0,n_pcs=100,gamma=0.5'."""
    if name in METHODS:
        return METHODS[name]
    if name.startswith("sr:"):
        kw = {}
        for item in filter(None, name[3:].split(",")):
            k, v = item.split("=")
            kw[k] = (bool(int(v)) if k == "dens_own" else
                     int(v) if k in {"n_hvg", "n_pcs", "n_epochs", "per_point", "n_neg", "kmax", "pool_per_level", "n_cand", "dens_k", "dens_all", "dens_pcs", "dens_refresh", "seed"} else float(v))
        return scalerank(**kw)
    if name.startswith("task:"):
        base, *items = name[5:].split(",")
        kw = dict(TASK_DEFAULTS[base])
        for item in items:
            k, v = item.split("=")
            kw[k] = float(v) if k == "gamma" else int(v)
        return task_method(base, **kw)
    raise KeyError(name)


METHODS = {
    "umap": m_umap,
    "tsne": m_tsne,
    "scalerank_g1": scalerank(gamma=1.0),
    "scalerank_g0.5": scalerank(gamma=0.5),
    "scalerank_g1_ns0.5": scalerank(gamma=1.0, neg_scale=0.5),
}


# ---------------------------------------------------------------- embedding processor (as in process_embedding/script.py)
def process(Y, solution):
    w = solution.obs["is_waypoint"].values
    out = {"between_waypoint_distances": scipy.spatial.distance.cdist(Y[w], Y[w])}  # no n x n temporaries
    np.fill_diagonal(out["between_waypoint_distances"], 0)
    labels = np.unique(solution.obs["cell_type"])
    cent = np.stack([Y[(solution.obs["cell_type"] == l).values].mean(axis=0) for l in labels])
    out["waypoint_centroid_distances"] = pairwise_distances(Y[w], cent)
    np.fill_diagonal(out["waypoint_centroid_distances"], 0)
    out["between_centroid_distances"] = pairwise_distances(cent, cent)
    np.fill_diagonal(out["between_centroid_distances"], 0)
    return out


# ---------------------------------------------------------------- metrics
def _ranks(D, rows):
    """Ranks within each row of D[rows] (self = 0), stable on ties, int32."""
    B = D[rows].astype(np.float64, copy=True)
    B[np.arange(len(rows)), rows] = -np.inf  # self gets rank 0
    order = np.argsort(B, axis=1, kind="stable")
    R = np.empty(order.shape, dtype=np.int32)
    np.put_along_axis(R, order, np.arange(D.shape[1], dtype=np.int32)[None, :], axis=1)
    return R


def row_ranks(D, chunk=256):
    """All row ranks of D as an n x n int32 matrix (built in row chunks)."""
    n = D.shape[0]
    R = np.empty((n, n), dtype=np.int32)
    for a in range(0, n, chunk):
        rows = np.arange(a, min(a + chunk, n))
        R[rows] = _ranks(D, rows)
    return R


def coranking(Rh, Dl, k=30, chunk=256):
    """coRanking metrics without the (n-1)^2 co-ranking matrix.

    Rh: row ranks of the high-dimensional distances (row_ranks), reused across
    methods. Q_NX(K) counts pairs with both ranks <= K, i.e. max(rank_h,
    rank_l) <= K, so a histogram of max ranks gives every K at once. T&C at k
    are sums over the off-diagonal blocks of Q, accumulated pair by pair.
    """
    n = Rh.shape[0]
    hist = np.zeros(n, dtype=np.int64)
    trust_s = cont_s = 0
    for a in range(0, n, chunk):
        rows = np.arange(a, min(a + chunk, n))
        rh_, rl_ = Rh[rows], _ranks(Dl, rows)
        off = rh_ > 0  # drop self
        rh, rl = rh_[off].astype(np.int64), rl_[off].astype(np.int64)
        hist += np.bincount(np.maximum(rh, rl), minlength=n)
        m = (rh > k) & (rl <= k)
        trust_s += int((rh[m] - k).sum())
        m = (rh <= k) & (rl > k)
        cont_s += int((rl[m] - k).sum())
    K = np.arange(1, n)
    qnx = np.cumsum(hist)[1:] / (K * n)
    lcmc = qnx - K / (n - 1)
    kmax = int(np.argmax(lcmc)) + 1
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return {
        "continuity_at_k30": 1 - norm * cont_s, "trustworthiness_at_k30": 1 - norm * trust_s,
        "qnx_at_k30": qnx[k - 1], "lcmc_at_k30": lcmc[k - 1],
        "qnx_auc": qnx.mean(), "qlocal": qnx[:kmax].mean(), "qglobal": qnx[kmax:].mean(),
    }


def _upper(D):
    """Strict upper triangle of D, row by row, in D's dtype."""
    n = D.shape[0]
    out = np.empty(n * (n - 1) // 2, dtype=D.dtype)
    p = 0
    for i in range(n - 1):
        out[p:p + n - 1 - i] = D[i, i + 1:]
        p += n - 1 - i
    return out


def full_ranks(D, pop=None):
    """Average ranks that the full symmetric matrix D (non-negative, zero
    diagonal) gives its values, as scipy's rankdata on D.ravel() would:
    returns (ranks of the upper-triangle values, rank of the diagonal zeros).
    Each off-diagonal value occurs twice in D, and the diagonal adds n zeros."""
    n = D.shape[0]
    tri = _upper(D)
    if pop is not None:  # free the n x n matrix once its triangle is copied
        del D
        pop()
    order = np.argsort(tri, kind="stable")
    srt = tri[order]
    del tri
    m = len(srt)
    nz = int(np.searchsorted(srt, 0, side="right"))  # off-diagonal zeros
    rs = np.arange(m, dtype=np.float64)
    rs *= 2
    rs += n + 1.5  # untied positive value at sorted position a: 2a + n + 3/2
    zero_rank = (2.0 * nz + n + 1) / 2
    rs[:nz] = zero_rank
    tie = np.flatnonzero(srt[1:] == srt[:-1])
    del srt
    tie = tie[tie >= nz]
    if len(tie):  # blocks of equal positive values: 2a + n + (2t + 1)/2
        first = np.r_[True, np.diff(tie) != 1]
        last = np.r_[first[1:], True]
        k0, k1 = tie[first], tie[last]
        t = k1 - k0 + 2
        offs = np.arange(int(t.sum())) - np.repeat(np.cumsum(t) - t, t)
        rs[np.repeat(k0, t) + offs] = np.repeat(2.0 * k0 + n + (2.0 * t + 1) / 2, t)
    r = np.empty(m, dtype=np.float64)
    r[order] = rs
    return r, zero_rank


def sym_spearman(high, proc, key):
    """spearmanr(Dh, Dl, axis=None) for symmetric, non-negative matrices with a
    zero diagonal; high = full_ranks(Dh), reused across methods."""
    rh, dh = high
    n = proc[key].shape[0]
    rl, dl = full_ranks(proc[key], pop=lambda: proc.pop(key))
    w = 2 * len(rh) + n
    mh = (2 * rh.sum() + n * dh) / w
    ml = (2 * rl.sum() + n * dl) / w
    rl -= ml
    cov = 2 * (np.dot(rh, rl) - mh * rl.sum()) + n * (dh - mh) * (dl - ml)
    vh = 2 * (np.dot(rh, rh) - 2 * mh * rh.sum() + len(rh) * mh * mh) + n * (dh - mh) ** 2
    vl = 2 * np.dot(rl, rl) + n * (dl - ml) ** 2
    return cov / np.sqrt(vh * vl)


def distance_correlations(proc, solution, high):
    """Last user of the embedding's waypoint distances: frees them."""
    s = lambda key: scipy.stats.spearmanr(solution.uns[key], proc[key], axis=None).correlation
    out = {"centroid_distance_correlation": s("waypoint_centroid_distances"),
           "label_distance_correlation": s("between_centroid_distances")}
    out["waypoint_distance_correlation"] = sym_spearman(high["spearman"], proc, "between_waypoint_distances")
    return out


def _radii(X, n_neighbors=30, random_state=42):
    # from density_preservation/script.py (after umap-learn, BSD 3-clause)
    from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors

    knn_indices, knn_dists, _ = nearest_neighbors(X, n_neighbors, "euclidean", {}, False, random_state, verbose=False)
    g, _, _, d = fuzzy_simplicial_set(X, n_neighbors, random_state, "euclidean", {}, knn_indices, knn_dists, verbose=False, return_dists=True)
    g = g.tocoo()
    g.sum_duplicates()
    g.eliminate_zeros()
    mu_sum = np.zeros(g.shape[1], dtype=np.float32)
    re = np.zeros(g.shape[1], dtype=np.float32)
    for j, k, mu in zip(g.row, g.col, g.data):
        D = d[j, k]
        re[j] += mu * D
        re[k] += mu * D
        mu_sum[j] += mu
        mu_sum[k] += mu
    return np.log(1e-8 + re / mu_sum)


def density_preservation(Y, solution, ro_cache={}):
    key = id(solution)
    if key not in ro_cache:
        ro_cache[key] = _radii(solution.layers["normalized"])
    return {"density_preservation": pearsonr(ro_cache[key], _radii(Y))[0]}


def clustering(Y, solution):
    a = ad.AnnData(obs=solution.obs[[]].copy())
    a.obsm["X_emb"] = Y
    sc.pp.neighbors(a, use_rep="X_emb")
    best, best_key = -1, None
    for res in [2 * x / 20 for x in range(1, 21)]:
        key = f"leiden_{res}"
        sc.tl.leiden(a, resolution=res, key_added=key, flavor="igraph", n_iterations=2, directed=False)
        sc_ = normalized_mutual_info_score(solution.obs["cell_type"], a.obs[key], average_method="arithmetic")
        if sc_ > best:
            best, best_key = sc_, key
    return {"normalized_mutual_information": best,
            "adjusted_rand_index": adjusted_rand_score(solution.obs["cell_type"], a.obs[best_key])}


def spectral_distance_correlation(Y, solution, cache={}):
    import umap
    import umap.spectral

    key = id(solution)
    if key not in cache:
        high = solution.layers["normalized"]
        n_comps = min(1000, min(solution.shape) - 2)
        graph = umap.UMAP(transform_mode="graph").fit_transform(high)
        cache[key] = scipy.spatial.distance.pdist(umap.spectral.spectral_layout(high, graph, n_comps, random_state=np.random.default_rng(0)))
    return {"spectral_distance_correlation": scipy.stats.spearmanr(scipy.spatial.distance.pdist(Y), cache[key]).correlation}


def main():
    data_dir, out_csv, *names = sys.argv[1:]
    names = names or list(METHODS)
    for name in names:
        method(name)  # fail early on a bad spec
    dataset = ad.read_h5ad(f"{data_dir}/dataset.h5ad")
    solution = ad.read_h5ad(f"{data_dir}/solution.h5ad")
    ds = solution.uns["dataset_id"].split("/")[-1]
    print(f"{ds}: {solution.n_obs} cells, {solution.n_vars} genes, {int(solution.obs['is_waypoint'].sum())} waypoints, {solution.obs['cell_type'].nunique()} cell types", flush=True)
    # the high-dimensional waypoint distances are only needed as ranks: compute
    # them once and drop the matrix
    Dh = solution.uns.pop("between_waypoint_distances")
    high = {"rows": row_ranks(Dh), "spearman": full_ranks(Dh)}
    del Dh
    rows = []
    for name in names:
        t = time.time()
        Y = np.asarray(method(name)(dataset), dtype=np.float64)
        secs = time.time() - t
        proc = process(Y, solution)
        scores = {"time_s": secs}
        scores |= coranking(high["rows"], proc["between_waypoint_distances"])
        scores |= distance_correlations(proc, solution, high)
        scores |= density_preservation(Y, solution)
        scores |= clustering(Y, solution)
        if not os.environ.get("SKIP_SPECTRAL"):  # pdist of all cells and 1000 spectral components: slow on large data
            scores |= spectral_distance_correlation(Y, solution)
        rows.append({"dataset": ds, "method": name, **scores})
        print(f"  {name:20s} {secs:6.1f}s  qnx_auc {scores['qnx_auc']:.3f}  qlocal {scores['qlocal']:.3f}  qglobal {scores['qglobal']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        del Y, proc  # the waypoint distance matrix is large


if __name__ == "__main__":
    main()
