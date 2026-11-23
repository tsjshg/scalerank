"""drbench methods for ScaleRank, loaded through the ``drbench.methods`` entry point.

    scalerank       the Python prototype with N x N rank tables (scalerank.py)
    scalerank_fast  the N^2-free Python version (scalerank_fast.py)
    scalerank_rs    the Rust version, via the scalerank package (crates/scalerank-py)
    tsne_rs         the Rust t-SNE (openTSNE's defaults, Barnes-Hut)
    umap_rs         the Rust UMAP (umap-learn's defaults, PCA initialisation)

The *_rs names are kept so earlier drbench results still match. tsne_rs and
umap_rs need the package built with the ``baselines`` feature.

Parameters go in the spec, e.g. ``scalerank:gamma=0.5``.
"""

import numpy as np
from drbench.methods import register


@register("scalerank")
def scalerank(X, seed=0, **kw):
    from .scalerank import ScaleRank

    return ScaleRank(random_state=seed, **kw).fit_transform(X)


@register("scalerank_fast")
def scalerank_fast(X, seed=0, **kw):
    from .scalerank_fast import ScaleRankFast

    return ScaleRankFast(random_state=seed, **kw).fit_transform(X)


@register("scalerank_rs")
def scalerank_rs(X, seed=0, **kw):
    import scalerank

    return scalerank.ScaleRank(random_state=seed, **kw).fit_transform(np.ascontiguousarray(X, dtype=np.float64))


@register("umap_rs")
def umap_rs(X, seed=0, **kw):
    from scalerank import _core

    return _core.umap(np.ascontiguousarray(X, dtype=np.float64), random_state=seed, **kw)


@register("tsne_rs")
def tsne_rs(X, seed=0, **kw):
    from scalerank import _core

    return _core.tsne(np.ascontiguousarray(X, dtype=np.float64), random_state=seed, **kw)
