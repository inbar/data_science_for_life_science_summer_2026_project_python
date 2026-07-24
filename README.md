# Benchmarking dependency measures for marker-gene identification in multi-modal single-cell data

## Introduction

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

## Repository Structure

```text
├── src/
│   ├── deep_learning/         # MLP training and tuning
│   ├── exploratory_analysis/  # Helper functions for plotting and dimentionality reduction
│   ├── measures/              # Spearman, Partial Corr, CMI, and MLP evaluators
│   └── preprocessing/         # Helper functions for preprocessing and normalization steps
│
├── scripts/                   # Entrypoint scripts to run full execution pipeline
├── cluster_scripts/           # Entrypoint scripts for the allegro cluster
├── notebooks/                 # Exploratory analysis and figure drafting
├── local_data/                # All results and small generated data to be committed to the repo
│
└── environment.yml            # Conda environment definition
```

## Quick links

* **Overleaf**
  * https://www.overleaf.com/project/6a16a6acee7a4f9be406ed56

* **Drive**  
  * [Drive Folder](https://drive.google.com/drive/folders/1EbHnfwc--__TVGd0T7sYfsFyHT2DoH8q)  
    * ➡️ [Pitch slides](https://docs.google.com/presentation/d/1NsWcmVj_nGgPXwznGEQuIt0szK6gWpAv6AhVMVyTu64)  
    * ➡️ [Project presentation slides](https://docs.google.com/presentation/d/1XC8spsQxBdkZpUi3c1FwNNoBYlBeGhgpELGiPLkxLGU)  


## Notebooks

Analysis was conducted accross two axes of granularity: 
1. The annotation level (how many different cell types were annontated)
    * L1 - 8 cell types
    * L2 - 30 cell types
2. The sample size
    * Full dataset (161k cells)
    * Subsamples: 10k, 25k

The analysis for the different combinations is available in the corresponding notebooks:

|| L1 | L2 | 
|--|--|--|
|10k|[Notebook](./notebooks/pipeline_validation_l1_10k.ipynb)|N/A|
|25k|[Notebook](./notebooks/pipeline_validation_l1_25k.ipynb)|[Notebook](./notebooks/pipeline_validation_l2_10k.ipynb)|
|Full dataset (161K)|[Notebook](./notebooks/pipeline_validation_l1_1161k.ipynb)|[Notebook*](./notebooks/pipeline_validation_l2_10k.ipynb)|

\* Note: Functional annotation (GO enrichment) was only conducted on the L2/161K dataset. 

## Run on cluster
See [cluster_scripts/README.md](./cluster_scripts/README.md)

## Run locally 

### 1. Download the data

Download the raw data archives and place them in the root of this repository. Subsequent scripts will look for them there. 

Required files: 
* `GSE164378_RAW.tar` ([download](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/GSE164378_RAW.tar)) 
  * Data matrices for all modalities in Matrix Market (.mtx) format 
  * ~1.4G
* `GSE164378_sc.meta.data_3P.csv.gz` ([download](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/GSE164378_sc.meta.data_3P.csv.gz))
  * Metadata
  * ~3M
* FTP Directory: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/ 

Alternatively:
```
cd <root_of_repository>

# GSE164378_RAW.tar
curl -o GSE164378_RAW.tar https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/GSE164378_RAW.tar

# GSE164378_sc.meta.data_3P.csv.gz
curl -o GSE164378_sc.meta.data_3P.csv.gz https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/GSE164378_sc.meta.data_3P.csv.gz
```

### 2. Run the pipeline

```bash
conda env create -f environment.yml
conda activate data_science_in_life_sciences_project_2026_group_1


$ ./run_validation_pipeline.py <parameters>

# To see all available parameters:
$ ./run_validation_pipeline.py -h
```
---

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

Always run through the activated env (`conda activate data_science_in_life_sciences_project_2026_group_1` or
`conda run -n data_science_in_life_sciences_project_2026_group_1 ...`). The conda-forge BLAS depends on DLLs in the
env's `Library\bin`, which is only on `PATH` when the env is activated.
