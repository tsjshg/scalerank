"""ScaleRank: dimensionality reduction that preserves neighbour ranks over a
range of scales, weighted by p(K) ~ K^-gamma, without N x N structures.

    >>> from scalerank import ScaleRank
    >>> Y = ScaleRank(gamma=0.5, random_state=0).fit_transform(X)

``X`` is a float array of shape (n_samples, n_features); ``Y`` has shape
(n_samples, n_components).
"""

from ._core import ScaleRank, __version__, evaluate, knn, rank_estimates

__all__ = ["ScaleRank", "__version__", "evaluate", "knn", "rank_estimates"]
