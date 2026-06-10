# Benchmarking dependency measures for marker-gene identification in multi-modal single-cell data

A 2×2 benchmark of dependency measures for identifying cell-type marker genes in
PBMC CITE-seq data (Hao et al. 2021), evaluated against a **protein-derived
ground truth** built from the ADT (surface-protein) modality.

|            | Marginal                     | Conditional                          |
|------------|------------------------------|--------------------------------------|
| **Linear**    | Spearman correlation         | Partial correlation (shrinkage)      |
| **Nonlinear** | Mutual information (KSG)      | Integrated Gradients on an MLP       |

The question: do *nonlinearity* and *multivariate context* change which genes are
called markers, and which axis matters more? All four methods see **RNA only**; the
protein modality is used solely to define the ground-truth driver set `D_c` per
cell type, keeping the evaluation a genuine cross-modal test.

## Repository layout

```
Data_Science.pdf            original project pitch
environment.yml             conda environment (env name: marker-bench)
src/                        all reusable logic (imported by the notebook)
  config.py                 paths, seeds, constants, method labels
  data_io.py                download/extract -> cached subsampled RNA+ADT MuData
  preprocessing.py          QC, normalization, HVGs, shared rank matrix, embeddings
  protein_gene_map.py       curated antibody -> encoding-gene map
  ground_truth.py           protein-derived driver sets D_c
  scorers.py                Spearman / partial-correlation / KSG-MI
  mlp.py                    tanh+softmax MLP + Integrated-Gradients attributions
  metric.py                 AUC_rel driver-recovery metric
  benchmark.py              run all methods × all cell types
  stats.py                  Friedman + paired Wilcoxon, bootstrap/seed stability
  plotting.py               publication figures (600 dpi, no titles)
notebooks/
  marker_benchmark.ipynb    the documented, end-to-end workflow
run_pipeline.py             staged CLI driver (used to validate / precompute)
build_cache.py              one-time heavy build of the subsampled dataset
data/                       raw download + processed cache (not version-controlled)
results/figures/            600 dpi PDF + PNG figures
results/tables/             CSV result tables
```

## Reproduce

```bash
conda env create -f environment.yml
conda activate marker-bench

# 1. download (~1.4 GB) + build the cached 25k-cell subsample (one-time)
python build_cache.py

# 2a. run the whole thing staged from the CLI ...
python run_pipeline.py all          # prep, gt, mlp, bench, stats
python run_pipeline.py plots        # all publication figures
python run_pipeline.py stability    # bootstrap-over-cells CIs
python run_pipeline.py seed         # MLP/IG seed variation

# 2b. ... or open the documented notebook (same src calls, with narrative)
jupyter lab notebooks/marker_benchmark.ipynb   # kernel: Python (marker-bench)
```

## Key design decisions

- **Data**: Hao 2021 PBMC CITE-seq from GEO `GSE164378` (3′ assay: RNA + 228 ADT +
  donor/`celltype.l1/l2/l3` metadata). CELLxGENE hosts only the RNA, so GEO is used
  to obtain both modalities. The full 161k×33k matrix is *streamed* to keep only a
  **sqrt-proportional stratified subsample** (~25k cells), which retains rare PBMC
  types (ILC, cDC1, HSPC) that pure proportional sampling would lose.
- **Shared feature space**: all four methods consume the *identical*
  rank-transformed, z-scored matrix of **HVGs ∪ protein-encoding marker genes**
  (~2.1k genes). The union guarantees every protein-derived driver gene is actually
  in the ranked set (no leakage — features stay RNA-only). The average-rank
  transform also collapses dropout zeros to a shared rank, mitigating the
  zero-inflation confound for MI.
- **Partial correlation**: the empirical gene covariance is ill-conditioned
  (dropout, collinearity, p≈n), so we use a **Ledoit–Wolf shrinkage** covariance →
  precision matrix → point-biserial partial correlation against the cell-type
  indicator. (The notebook shows the eigen-spectrum that motivates this.)
- **MLP**: `tanh` hidden layers + `softmax` output (predictions on the simplex),
  cross-entropy with inverse-frequency class weights, early stopping on validation
  loss; Integrated Gradients (Captum) attributions per gene per class.
- **Ground truth**: per cell type, the top-`k` ADT proteins by one-vs-rest Wilcoxon
  *score* (p-values are uninformative at n≈25k), mapped to encoding genes by
  molecular fact only. Cell types with `|D_c| < 2` are excluded.
- **Metric**: `AUC_rel`, the normalized driver-recovery AUC (≡ Mann–Whitney ROC-AUC
  of the per-gene score discriminating drivers from non-drivers; 0.5 = random,
  1.0 = perfect), parameter-free and defined on every method's output.
- **Statistics**: the unit of replication is the **cell type** (paired across
  methods) — Friedman omnibus + Holm-corrected pairwise Wilcoxon signed-rank.
  Bootstrap-over-cells and MLP-seed variation are reported as descriptive
  stability bands, not inference.

## Environment note (Windows)

Always run through the activated env (`conda activate marker-bench` or
`conda run -n marker-bench ...`). The conda-forge BLAS depends on DLLs in the
env's `Library\bin`, which is only on `PATH` when the env is activated.
