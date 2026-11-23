# ScaleRank on Open Problems (dimensionality reduction)

ScaleRank scored on the four `openproblems_v1` datasets of
[task_dimensionality_reduction](https://github.com/openproblems-bio/task_dimensionality_reduction),
next to the published results of run `2024-12-19_15-35-59`. Runs of
2026-11-22. These are local re-implementations of the task's pipeline, not
runs through Viash / Nextflow, so the numbers are close to, but not the same
as, what an official submission would give.

| File | Contents |
|---|---|
| `tables.md` | Every metric × every method, per dataset: our runs `(ours)` first, then the published methods and the controls |
| `summary.csv` | The same in long form (dataset, metric, method, source, value). For our runs, `rank` is the rank among the published non-control methods plus that run (1 = best) |
| `runs/<dataset>.csv` | Our raw scores, including `time_s` |
| `published_scores.csv` | The published scores for these four datasets, parsed from the run's `score_uns.yaml` |
| `harness/op_eval.py` | Data preparation, methods, embedding processor and metrics |
| `harness/compare.py` | Ours vs published, one dataset |
| `harness/summarize.py` | Builds `summary.csv` and `tables.md` from the files above |
| `runs/sweep/<dataset>.csv` | Input × γ sweep (see below) |
| `runs/hvg/<dataset>.csv` | The task's baselines with their default input and with 3,000 HVGs |
| `runs/dens/<dataset>.csv` | ScaleRank with the density term (see below); one row per setting |
| `harness/sweep_summary.py` | Overall score (Open Problems style) of our runs next to the published methods |
| `runs/seeds/*.csv` | Extra seeds (1–4 for ScaleRank, 1–4 or 1–2 for densMAP) of the density settings (see below) |
| `harness/seed_summary.py` | Overall score per setting, mean ± sd over seeds |

## Setup

- **Data:** the task's `log_cp10k` `dataset.h5ad` and `solution.h5ad`, from the public bucket
  `s3://openproblems-data/resources/task_dimensionality_reduction/datasets/openproblems_v1/<dataset>/log_cp10k/`.

  | Dataset | Cells | Cell types |
  |---|---:|---:|
  | mouse_blood_olsson_labelled | 660 | 7 |
  | mouse_hspc_nestorowa2016 | 1,920 | 8 |
  | pancreas | 16,382 | 14 |
  | zebrafish | 26,022 | 24 |

  Every cell is a waypoint.
- **Methods:** as in the task. Input is `layers['normalized']`, top 1,000 genes by `var['hvg_score']`.
  - `umap`: umap-learn on the HVGs, `random_state=42`. This is the task's default.
  - `tsne`: scanpy t-SNE on PCA-50 of the 1,000 HVGs. This is **not** the task's default, which uses all genes. That run is `task:tsne` in `runs/hvg/`.
  - `scalerank_g1`: `scalerank.ScaleRank(gamma=1.0, random_state=42)` on PCA-50 of the HVGs (`sc.tl.pca`, arpack).
  - `scalerank_g1_ns0.5`: the same with `neg_scale=0.5`.
  - `scalerank_g0.5`: the same with `gamma=0.5`. Small datasets only.
- **Metrics:**
  - The co-ranking metrics (qnx_auc, qlocal, qglobal, Q_NX and LCMC at K = 30, trustworthiness and continuity at K = 30) are re-implemented from the definitions of the R coRanking package. They are computed on the waypoint distance matrices without building the co-ranking matrix. They agree exactly with a direct implementation through the full matrix.
  - The distance correlations are Spearman over the full waypoint, centroid and label distance matrices. The waypoint one uses an exact upper-triangle form of `spearmanr(axis=None)`.
  - Density preservation, Leiden NMI/ARI and spectral distance correlation follow the task's Python scripts.
  - `spectral_distance_correlation` was skipped on pancreas and zebrafish because of its cost.
- **Where it ran:**
  - olsson and nestorowa on an M4 Mac.
  - pancreas and zebrafish on a Linux box: i7-8700B, 12 threads, Ubuntu 24.04, peak memory 21 GB and 24 GB.
  - Times are wall-clock on those machines and are not comparable across the two groups.

**Harness check.** Our `umap` reproduces the published `umap` within noise:

| Dataset | qnx_auc (ours) | qnx_auc (published) | NMI (ours) | NMI (published) |
|---|---:|---:|---:|---:|
| olsson | 0.637 | 0.639 | 0.668 | 0.671 |
| pancreas | 0.598 | 0.599 | 0.669 | 0.672 |

Our `tsne` differs more from the published one, mainly because it used 1,000 HVGs instead of all genes. With all genes (`task:tsne`), olsson qnx_auc is 0.659, against 0.679 published. Our PCA with the task's default input reproduces the published value exactly (0.623).

## Results

ScaleRank γ = 1: value and rank among the published non-control methods plus
ScaleRank. The denominators differ because some published methods lack some
metrics, and the published zebrafish `umap` has no co-ranking scores.

| Metric | olsson | nestorowa | pancreas | zebrafish |
|---|---|---|---|---|
| qnx_auc | 0.700 (3/12) | 0.517 (10/11) | 0.615 (5/9) | 0.626 (5/7) |
| qlocal | 0.503 (4/12) | 0.164 (7/11) | 0.254 (4/9) | 0.267 (5/7) |
| qglobal | 0.726 (2/12) | 0.562 (10/11) | 0.636 (5/9) | 0.664 (6/7) |
| trustworthiness@30 | 0.971 (4/12) | 0.645 (4/11) | 0.866 (3/9) | 0.879 (4/7) |
| continuity@30 | 0.968 (5/12) | 0.591 (8/11) | 0.841 (4/9) | 0.818 (6/7) |
| waypoint distance corr. | 0.399 (7/12) | −0.049 (9/11) | 0.309 (4/11) | 0.387 (5/9) |
| density preservation | 0.666 (3/11) | 0.113 (6/10) | 0.243 (3/9) | 0.093 (2/9) |
| NMI | 0.661 (5/11) | 0.210 (5/11) | 0.655 (3/11) | 0.551 (2/9) |
| ARI | 0.317 (6/11) | 0.096 (10/11) | 0.272 (4/11) | 0.165 (7/9) |

Run time in seconds (same machine within a dataset):

| | olsson | nestorowa | pancreas | zebrafish |
|---|---:|---:|---:|---:|
| umap | 7.4 | 8.1 | 73.0 | 67.1 |
| tsne | 0.9 | 3.3 | 48.6 | 76.7 |
| scalerank_g1 | 0.4 | 1.1 | 23.5 | 39.6 |

### Reading

- **Against UMAP**, run in the same harness:
  - ScaleRank is higher on waypoint distance correlation on all four datasets.
  - It is higher on qnx_auc, qglobal and density preservation on three of the four. On nestorowa it is slightly below UMAP (qnx_auc 0.517 vs 0.521).
  - It is 1.7 to 18 times faster. UMAP runs on the 1,000 HVGs, ScaleRank on their PCA-50.
- **Among all published methods** it is mid-table. The co-ranking leaders are different methods: PHATE on pancreas (qnx_auc 0.676), PCA, PHATE and PyMDE on zebrafish (0.687, 0.680, 0.671), and PHATE and t-SNE on nestorowa.
- **On the large datasets** the global co-ranking scores favour methods that keep the coarse layout (PCA, PHATE). This is where ScaleRank's γ = 1 setting, which leans local, loses most.
- **neg_scale = 0.5** and **γ = 0.5** were not better than the defaults.

## Input and γ sweep

The results above use one fixed setting: 1,000 HVG → PCA-50, γ = 1. A
sweep of the input and of γ followed. It ran on the Linux box, in
`runs/sweep/`. The method spec is `sr:n_hvg=…,n_pcs=…,gamma=…`, where
`n_hvg=0` means all genes.

1. **Grid:** 18 settings on the two small datasets: n_hvg ∈ {1000, 3000, all} × n_pcs ∈ {50, 100} × γ ∈ {0, 0.5, 1}.
2. **Choice:** three candidates chosen from those results.
3. **Check:** the candidates run on pancreas and zebrafish.

**Overall score.** `harness/sweep_summary.py` computes it as in Open
Problems' overall ranking:

- each metric is scaled to [0, 1] between the worst and best control and clipped;
- the scaled metrics are averaged;
- a missing metric counts as 0;
- spectral distance correlation is left out.

| Setting / method | olsson | nestorowa | pancreas | zebrafish | mean |
|---|---:|---:|---:|---:|---:|
| ScaleRank 3000 HVG → PCA-100, γ = 0.5 | 0.592 | 0.328 | 0.370 | 0.459 | **0.437** |
| ScaleRank 3000 HVG → PCA-100, γ = 1 | 0.614 | 0.331 | 0.328 | 0.463 | 0.434 |
| ScaleRank all genes → PCA-100, γ = 1 | 0.531 | 0.340 | 0.333 | 0.512 | 0.429 |
| PHATE (best published, mean) | 0.514 | 0.288 | 0.400 | 0.438 | 0.410 |
| ivis | 0.592 | 0.264 | 0.358 | 0.415 | 0.407 |
| ScaleRank 1000 HVG → PCA-50, γ = 1 (first runs) | 0.579 | 0.243 | 0.355 | 0.429 | 0.401 |
| neuralee | 0.585 | 0.282 | 0.335 | 0.403 | 0.401 |

qnx_auc of the same settings:

| Setting | olsson | nestorowa | pancreas | zebrafish |
|---|---:|---:|---:|---:|
| 3000 → 100, γ = 0.5 | 0.698 | 0.562 | 0.638 | 0.645 |
| 3000 → 100, γ = 1 | 0.700 | 0.557 | 0.609 | 0.636 |
| all → 100, γ = 1 | 0.643 | 0.571 | 0.622 | 0.687 |
| 1000 → 50, γ = 1 | 0.697 | 0.517 | 0.615 | 0.626 |
| best published | 0.701 (densmap) | 0.569 (phate) | 0.676 (phate) | 0.687 (pca) |

### Reading the sweep

- **The input matters more than γ.** On the small datasets γ ∈ {0, 0.5, 1} moves qnx_auc by about 0.01 to 0.02. The gene set and the number of PCs move it by up to 0.07.
- **Matching the evaluation space helps when the data allow it.** The reference distances are on all genes. All genes → PCA-100 is best on nestorowa and zebrafish (zebrafish qnx_auc 0.687, level with PCA) but clearly worse on olsson, which has 116,918 features. 3,000 HVG → PCA-100 is the most even.
- **PHATE stays ahead on pancreas.** Its overall score there is 0.400, against 0.370 for ScaleRank's best setting.

### Caveats

- **The settings were chosen on the same benchmark.** They were picked on the two small datasets and checked on the two large ones, but all four are the benchmark's own data.
- **One seed** (`random_state=42`). The same setting differs by about 0.003 in qnx_auc between the Mac and the Linux box. The density settings were later rerun with more seeds (see *Seeds*).
- **Missing metrics count as 0.** This lowers the published methods that lack some metrics: densmap, lmds, and umap on zebrafish.
- **The spectral metric is left out.** On the small datasets ScaleRank was weakest there (olsson −0.024, last).
- **These four datasets only.** The published run also has the CELLxGENE datasets, which were not run here.

## Other methods with 3,000 HVGs

Five of the task's baselines were rerun in this harness: `task:<method>`
with the method's own default input, and `task:<method>,n_hvg=3000`. This
puts every method under the same conditions: one machine, the same metrics,
and no missing metrics. The results are in `runs/hvg/`.

The defaults come from `src/methods/*/config.vsh.yaml`:

| Method | Default input |
|---|---|
| UMAP | 1000 HVG, no PCA |
| t-SNE | all genes → PCA-50 |
| PCA | all genes |
| densMAP | all genes, no PCA |
| PHATE | all genes → PCA-50 |

**Overall score** (scaled as above, mean of the four datasets):

| Method | Default | 3000 HVG |
|---|---:|---:|
| densMAP | **0.457** | 0.421 |
| ScaleRank (1000 → 50 / 3000 → 100, γ = 1) | 0.406 | 0.434 |
| t-SNE | 0.412 | 0.410 |
| UMAP | 0.373 | 0.387 |
| PHATE (log_cp10k) | 0.358 | 0.388 |
| PCA | 0.343 | 0.383 |

**qnx_auc:**

| Method | olsson | nestorowa | pancreas | zebrafish |
|---|---|---|---|---|
| densMAP default → 3000 | 0.679 → 0.622 | 0.547 → 0.560 | 0.626 → 0.609 | 0.663 → 0.619 |
| t-SNE | 0.665 → 0.667 | 0.558 → 0.556 | 0.602 → 0.604 | 0.656 → 0.623 |
| UMAP | 0.637 → 0.632 | 0.521 → 0.550 | 0.598 → 0.618 | 0.612 → 0.625 |
| PHATE | 0.666 → 0.671 | 0.524 → 0.525 | 0.609 → 0.626 | 0.655 → 0.658 |
| PCA | 0.623 → 0.697 | 0.524 → 0.536 | 0.631 → 0.602 | 0.687 → 0.602 |

### What this changes

- **densMAP with its default input (all genes, no PCA) has the highest overall score here.** Its published scores lack the co-ranking metrics on the large datasets, so the published table counted them as 0. ScaleRank's lead over the published methods in the sweep table above is partly that artefact.
- **3,000 HVGs is not a general improvement.** It helps UMAP, PHATE and ScaleRank. It hurts densMAP. For PCA it helps on the small datasets and hurts on the large ones. For t-SNE it makes no clear difference.
- **The harness reproduces two published values on zebrafish:** PCA (0.687) and t-SNE (0.656 vs 0.657). PHATE does not reproduce (pancreas 0.609 vs 0.676 published), probably because the published PHATE used its preferred `sqrt_cp10k` normalisation, which is not in these files.
- **densMAP's default input is slow:** 23 to 27 minutes per large dataset here.

## Density term

The gap to densMAP above (0.457 vs 0.438 for 3000 → 100, γ = 0.5) is all in
`density_preservation`. Scaled per metric and averaged over the four
datasets, densMAP leads by 0.34 on that metric alone. That is 0.026 of the
overall score, more than the whole gap; the other metrics roughly cancel
(ScaleRank ahead on NMI/ARI, trustworthiness and waypoint correlation,
densMAP on continuity and label correlation). A rank loss is invariant to
the local scale, so it keeps no density information by construction.

`dens_lambda` > 0 adds densMAP's idea to ScaleRank: over the last
`dens_frac` (0.3) of the epochs, maximise the Pearson correlation between
local log radii r_i = ln(mean_j |y_i − y_j|²) in the embedding and a target,
over the input's `dens_k` (30) nearest neighbours; loss −λ · n · corr. It
costs O(n · k) per epoch. The target is measured in the input by default,
or passed as `fit_transform(x, dens_target=...)`. In the harness,
`dens_all=1` takes it from PCA-100 of all genes, the space the metric
compares to, while the ranks still come from 3,000 HVGs. The input's radii
track the metric's only partly: Pearson 0.78 / 0.86 (olsson / nestorowa)
for 3000 → 100, 0.90 / 0.95 for all genes → 100.

λ was chosen on the two small datasets (1–16 tried, 4–8 is a plateau, 16
is worse; `dens_frac` = 0.6 is no better). The large datasets are a held-out
check. One seed (42).

**Overall score** (3000 → 100, γ = 0.5):

| Method | olsson | nestorowa | pancreas | zebrafish | mean | time, large |
|---|---:|---:|---:|---:|---:|---:|
| ScaleRank + density, λ = 4, all-genes target | 0.616 | 0.420 | **0.409** | **0.472** | **0.479** | 74–90 s |
| ScaleRank + density, λ = 8, all-genes target | 0.612 | **0.428** | 0.406 | 0.461 | 0.477 | |
| ScaleRank + density, λ = 2, all-genes target | 0.615 | 0.398 | 0.408 | **0.474** | 0.474 | |
| ScaleRank + density, λ = 4, input target | 0.584 | 0.384 | 0.409 | 0.454 | 0.458 | 22–41 s |
| densMAP (task default, same harness) | **0.655** | 0.327 | 0.378 | 0.468 | 0.457 | 23–27 min |
| ScaleRank, no density term | 0.597 | 0.327 | 0.370 | 0.459 | 0.438 | 33–44 s |

**Key metrics, λ = 4 with the all-genes target** (before → after; densMAP in brackets):

| | olsson | nestorowa | pancreas | zebrafish |
|---|---|---|---|---|
| density_preservation | 0.605 → 0.841 (0.876) | 0.325 → 0.773 (0.651) | 0.238 → 0.572 (0.720) | 0.170 → 0.379 (0.779) |
| qnx_auc | 0.699 → 0.710 (0.679) | 0.562 → 0.603 (0.547) | 0.638 → 0.646 (0.626) | 0.645 → 0.648 (0.663) |

### Reading

- **With the density term, ScaleRank's overall score is above densMAP's
  (0.479 vs 0.457)**, on three of four datasets, in about 1/18 of densMAP's time
  on the large ones. olsson is still densMAP's.
- **It helps the other metrics too:** qnx_auc and qglobal go up on all four.
- **The target matters.** Radii from the HVG input lower density_preservation
  on olsson (0.605 → 0.577) and raise it less than the all-genes target on
  nestorowa and zebrafish; only on pancreas are they better (0.606 vs 0.572).
  Most of the extra time is the PCA of all genes.
- **The large datasets lagged densMAP on density** (0.57 / 0.38 vs 0.72 /
  0.78). The next section finds the two causes.

### Density on the large datasets

Diagnosis, from the radii the term pulls on to the ones the metric measures:

| | pancreas | zebrafish |
|---|---:|---:|
| target (all genes → PCA-100) vs the metric's reference radii | 0.668 | |
| target (all genes → PCA-300) vs reference | 0.887 | 0.824 |
| target (all genes, exact 30-NN, no PCA) vs reference | 0.938 | |
| λ = 4: embedding radii over the input's neighbours vs target | 0.989 | 0.979 |
| λ = 4: embedding radii over the embedding's own neighbours vs target | 0.944 | 0.515 |

The optimisation reaches its target (≈ 0.98), so two things were left:

1. **The target.** PCA-100 of all genes loses much of the local scale on the
   large datasets. PCA-300 (`dens_pcs=300`) or exact neighbours on the sparse
   all-genes matrix (`dens_pcs=0`) fix most of it.
2. **The neighbour set (zebrafish).** The metric measures each point's radius
   over its own 30 nearest neighbours *in the embedding*. On zebrafish these
   overlap little with the input's, so radii pulled into shape over the input's
   neighbours are not the ones the metric sees. `dens_own=True` measures the
   embedding's radii over its own nearest neighbours, recomputed every
   `dens_refresh` (10) epochs of the density phase (exact kNN in 2-D).

**density_preservation** (3000 → 100, γ = 0.5, λ = 4, all-genes target):

| Target / neighbours | olsson | nestorowa | pancreas | zebrafish |
|---|---:|---:|---:|---:|
| no density term | 0.588 | 0.327 | 0.238 | 0.170 |
| PCA-100 / input's | 0.841 | 0.773 | 0.572 | 0.379 |
| PCA-300 / input's | 0.833 | 0.818 | 0.773 | 0.490 |
| exact / input's | 0.852 | 0.811 | 0.828 | 0.425 |
| PCA-300 / own (`dens_own`) | 0.859 | **0.932** | **0.858** | 0.773 |
| densMAP | **0.876** | 0.651 | 0.720 | **0.779** |

**Overall score:**

| Setting | olsson | nestorowa | pancreas | zebrafish | mean | time, large (Linux) |
|---|---:|---:|---:|---:|---:|---:|
| PCA-300 / input's | 0.619 | **0.432** | **0.429** | 0.481 | **0.490** | 200–270 s |
| PCA-300 / own | 0.626 | 0.418 | 0.413 | **0.483** | 0.485 | 240–270 s |
| exact / input's | 0.615 | 0.426 | 0.425 | 0.474 | 0.485 | 390–690 s |
| PCA-100 / input's | 0.616 | 0.420 | 0.409 | 0.472 | 0.479 | 74–90 s |
| PCA-300 / own, λ = 8 | 0.575 | 0.431 | 0.376 | 0.463 | 0.461 | |
| densMAP | **0.655** | 0.327 | 0.378 | 0.468 | 0.457 | 23–27 min |
| no density term | 0.592 | 0.328 | 0.370 | 0.459 | 0.437 | 33–44 s |

- **`dens_own` closes the density gap**: equal to densMAP on zebrafish, above
  it on nestorowa and pancreas. But it costs NMI/ARI (pancreas NMI 0.644 →
  0.548) and some continuity, so the overall score does not rise: density is 1
  of the 13 metrics.
- **The best overall is PCA-300 with the input's neighbours, 0.490**, vs
  densMAP's 0.457, at 1/5 to 1/8 of its time. The next two rows are within
  0.005 of it at seed 42; the seeds below separate them.
- **Most of the extra time is the PCA of all genes** (arpack, 300
  components). The 2-D kNN refreshes of `dens_own` are brute force, O(n²)
  each; larger data will need a grid or a tree there.
- Not yet tried: λ below 4 with `dens_own`, a mix of the input's
  and the embedding's neighbours, and a faster all-genes PCA (randomised SVD).

### Seeds

The four main density settings, ScaleRank without the term, and densMAP,
rerun with other seeds (`seed=` in the spec; the input, the PCA and the
target do not depend on it). ScaleRank: seeds 42 and 1–4 on all four
datasets. densMAP: 42 and 1–4 on the small datasets, 42, 1 and 2 on the large
ones (about 25 min per run). `harness/seed_summary.py` averages over the seeds
that cover all four datasets.

**Overall score**, mean ± sd over seeds:

| Setting | olsson | nestorowa | pancreas | zebrafish | mean | seeds |
|---|---:|---:|---:|---:|---:|---:|
| PCA-300 / input's | 0.620 ± 0.003 | **0.430 ± 0.006** | **0.433 ± 0.004** | **0.481 ± 0.005** | **0.491 ± 0.002** | 5 |
| PCA-100 / input's | 0.617 ± 0.003 | 0.418 ± 0.004 | 0.414 ± 0.004 | 0.472 ± 0.008 | 0.480 ± 0.003 | 5 |
| PCA-300 / own | 0.617 ± 0.010 | 0.411 ± 0.014 | 0.412 ± 0.005 | 0.477 ± 0.010 | 0.479 ± 0.006 | 5 |
| densMAP | **0.633 ± 0.020** | 0.320 ± 0.006 | 0.383 ± 0.013 | 0.473 ± 0.008 | 0.452 ± 0.005 | 3 |
| no density term | 0.592 ± 0.004 | 0.327 ± 0.003 | 0.376 ± 0.004 | 0.459 ± 0.004 | 0.439 ± 0.001 | 5 |

**density_preservation**, mean ± sd over seeds:

| Setting | olsson | nestorowa | pancreas | zebrafish |
|---|---:|---:|---:|---:|
| PCA-300 / input's | 0.836 ± 0.009 | 0.801 ± 0.019 | 0.782 ± 0.005 | 0.493 ± 0.014 |
| PCA-100 / input's | 0.844 ± 0.010 | 0.755 ± 0.016 | 0.578 ± 0.006 | 0.378 ± 0.010 |
| PCA-300 / own | 0.866 ± 0.012 | **0.900 ± 0.029** | **0.858 ± 0.003** | 0.772 ± 0.003 |
| densMAP | **0.870 ± 0.007** | 0.658 ± 0.008 | 0.711 ± 0.009 | **0.778 ± 0.002** |
| no density term | 0.567 ± 0.014 | 0.325 ± 0.024 | 0.258 ± 0.022 | 0.176 ± 0.009 |

- **Seeds move the overall score by 0.001–0.006 (sd).** This is about the
  Mac vs Linux difference noted above.
- **PCA-300 with the input's neighbours is the best setting**: 0.491, 0.011
  above the next two, which cannot be told apart (0.480 and 0.479; `dens_own`
  also varies more between seeds).
- **It beats densMAP by 0.039** (0.491 vs 0.452): clearly on nestorowa and
  pancreas, narrowly on zebrafish (0.481 vs 0.473, within the seed spread).
  densMAP is ahead on olsson, where it also varies most (±0.020).
- **`dens_own` keeps the best density on nestorowa and pancreas and ties
  densMAP on zebrafish** over seeds too, but the overall ranking does not
  change.

## Reproducing

```bash
# Python 3.12 with: anndata scanpy umap-learn pynndescent leidenalg igraph scikit-learn
# plus scalerank (uv pip install ../../../crates/scalerank-py)
python harness/op_eval.py DATA/<dataset> runs/<dataset>.csv umap tsne scalerank_g1 scalerank_g1_ns0.5
SKIP_SPECTRAL=1 python harness/op_eval.py ...    # skips the spectral metric
python harness/summarize.py
```
