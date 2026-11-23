# scalerank

Python package of [ScaleRank](https://github.com/tsjshg/scalerank), a
dimensionality reduction that preserves neighbour ranks over a range of
scales. The implementation is in Rust.

```python
from scalerank import ScaleRank

Y = ScaleRank(gamma=0.5, random_state=0).fit_transform(X)   # X: (n, d) float array
```

See the repository's README for the method and the parameters.
