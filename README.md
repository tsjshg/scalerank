# ScaleRank

[日本語](README_ja.md)

A dimensionality reduction that preserves neighbour **ranks** over a range of
scales. For each point it samples a neighbourhood size K with probability
proportional to K^−γ, one point inside that neighbourhood and several outside
it, and moves the map so the inside point ends up closer. γ = 0 weights all
sizes equally and favours the global layout; larger γ concentrates on small
neighbourhoods and favours local detail. Exact ranks come from each point's
nearest neighbours and larger ranks are estimated from random subsamples, so
nothing is N × N.

An optional density term, after densMAP, also matches each point's local
radius to the data's; rank losses alone are blind to local scale.

- Browser app: <https://tsjshg.github.io/scalerank-web/> ([source](https://github.com/tsjshg/scalerank-web))
- Research notes: <https://tsjshg.github.io/scalerank-web/docs/>
- Benchmark: [drbench](https://github.com/tsjshg/drbench); results on Open Problems: [research/results/openproblems](research/results/openproblems/README.md)

## Python

```bash
pip install scalerank
```

```python
from scalerank import ScaleRank

Y = ScaleRank(gamma=0.5, random_state=0).fit_transform(X)   # X: (n, d) float array
```

Main parameters (all keyword arguments of `ScaleRank`):

| Parameter | Default | Meaning |
|---|---:|---|
| `gamma` | 1.0 | Scale exponent: p(K) ∝ K^−γ |
| `n_components` | 2 | Output dimensions |
| `n_epochs` | 800 | Optimisation epochs |
| `init` | `"pca"` | `"pca"` or `"random"` |
| `kmax` | 100 | Exact neighbours per point; larger ranks are estimated |
| `dens_lambda` | 0.0 | Weight of the density term (0 = off; 4 works on single-cell data) |
| `dens_k` | 30 | Neighbours that define a local radius |
| `dens_own` | False | Measure the map's radii over its own neighbours instead of the input's |
| `random_state` | None | Seed |

`fit_transform(X, dens_target=None)` takes optional target log radii, one per
row, when the density should follow another space than `X`. The package also
exports `knn`, `rank_estimates` and `evaluate` (the quality metrics).

## Repository layout

| Path | Contents |
|---|---|
| `crates/scalerank` | The method in Rust; its only dependency is the optional rayon (feature `parallel`), so it builds for WASM. kNN (brute force, NN-descent), PCA, N²-free rank tables, the optimiser and quality metrics |
| `crates/scalerank-py` | The Python package `scalerank` (PyO3, maturin). Feature `baselines` adds the Rust UMAP and t-SNE for the research scripts |
| `crates/scalerank-baselines` | UMAP and Barnes-Hut t-SNE on the same kNN and PCA, for comparison in the app and the research; not published |
| `research` | Python prototypes, experiment scripts, the research notes' source and the Open Problems results, as a drbench plugin (see [research/README.md](research/README.md)) |

## Development

```bash
cargo test --workspace
cd crates/scalerank-py && maturin develop --release              # the package
cd crates/scalerank-py && maturin develop --release --features baselines   # with UMAP / t-SNE
```
