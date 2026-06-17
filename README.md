# ️WIP: code cleanup 
1. Data download and dataset loading (`data.py`)  ✅
2. Create a non-generated notebook (based on the existing ones) 
3. Refactor and clean up .py scripts one by one

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

## Quick links

* **Overleaf**
  * https://www.overleaf.com/project/6a16a6acee7a4f9be406ed56

* **Drive**  
  * [Drive Folder](https://drive.google.com/drive/folders/1EbHnfwc--__TVGd0T7sYfsFyHT2DoH8q)  
    * ➡️ [Pitch slides](https://docs.google.com/presentation/d/1NsWcmVj_nGgPXwznGEQuIt0szK6gWpAv6AhVMVyTu64)  
    * ➡️ [Project presentation slides](https://docs.google.com/presentation/d/1XC8spsQxBdkZpUi3c1FwNNoBYlBeGhgpELGiPLkxLGU)  

## Reproduce

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

### 2. Run the code

```bash
conda env create -f environment.yml
conda activate data_science_in_life_sciences_project_2026_group_1

# 1. Load the raw data and save a 25k-cell subsample to disk 
# This should be run once. Subsequent runs load the data from disk.  
python build_dataset.py
python build_dataset.py true        # force recreation of the dataset

# it is po

# 2a. run the whole thing staged from the CLI ...
python run_pipeline.py all          # prep, gt, mlp, bench, stats
python run_pipeline.py plots        # all publication figures
python run_pipeline.py stability    # bootstrap-over-cells CIs
python run_pipeline.py seed         # MLP/IG seed variation

# 2b. ... or open the documented notebook (exact same procedure, with narrative)
jupyter lab notebooks/marker_benchmark.ipynb   # kernel: Python (data_science_in_life_sciences_project_2026_group_1)
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

Always run through the activated env (`conda activate data_science_in_life_sciences_project_2026_group_1` or
`conda run -n data_science_in_life_sciences_project_2026_group_1 ...`). The conda-forge BLAS depends on DLLs in the
env's `Library\bin`, which is only on `PATH` when the env is activated.
