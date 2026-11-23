# scalerank

**Maps you can read at every scale: the neighbourhoods inside each cluster,
and how the clusters sit relative to each other.**

[ScaleRank](https://github.com/tsjshg/scalerank) is a dimensionality reduction
that preserves neighbour ranks over a range of scales. The implementation is
in Rust.

- **Not beaten on both counts.** On the 8 datasets of
  [drbench](https://github.com/tsjshg/drbench), no other method was better on
  both local and global quality. This includes t-SNE and UMAP tuned over 30
  settings.
- **Global layout, kept.** The distance correlation between data and map is
  0.79, against 0.52 for t-SNE and 0.37 for UMAP.
- **A layout from the data, not the seed.** The global layout barely changes
  with the random seed.
- **Density, optionally.** A density term, after densMAP, also keeps how
  tightly points pack.

```python
from scalerank import ScaleRank

Y = ScaleRank(gamma=0.5, random_state=0).fit_transform(X)   # X: (n, d) float array
```

See the [repository](https://github.com/tsjshg/scalerank) for the method, the
parameters and the benchmarks.
