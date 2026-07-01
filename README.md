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

## Quick links

* **Overleaf**
  * https://www.overleaf.com/project/6a16a6acee7a4f9be406ed56

* **Drive**  
  * [Drive Folder](https://drive.google.com/drive/folders/1EbHnfwc--__TVGd0T7sYfsFyHT2DoH8q)  
    * ➡️ [Pitch slides](https://docs.google.com/presentation/d/1NsWcmVj_nGgPXwznGEQuIt0szK6gWpAv6AhVMVyTu64)  
    * ➡️ [Project presentation slides](https://docs.google.com/presentation/d/1XC8spsQxBdkZpUi3c1FwNNoBYlBeGhgpELGiPLkxLGU)  

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

### 2. Run the code

```bash
conda env create -f environment.yml
conda activate data_science_in_life_sciences_project_2026_group_1

## TODO!
```

## Run on cluster
The point of running stuff on the cluster is to  
(1) run intensive steps on the full data in parallel on powerful machines and   
(2) not clog the local disk with many GB of intermediate files.  

**Principally:**
1. Initial Setup Steps
    1. Clone git repo
    2. Setup conda
    3. Create datasets for later
1. Run intensive work
    1. Model training
    2. Scoring
1. Download the results for downstream analysis

The work for the cluster is provided in scripts, not notebooks.   
Analysis should be done somewhere else (locally, in a notebook). 
   
**Specifically:**

```bash
####
# Setup steps
#
# This should all be done on the login node and not sent as a job to the computing nodes. 
###
# SSH to the cluster (Allegro)
ssh -A <username>@allegro.imp.fu-berlin.de

###
# Clone git repo
###
$ mkdir workspace
$ cd workspace
$ git clone git@github.com:inbar/data_science_for_life_science_summer_2026_project_python.git

###
# Conda
#
# See: https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install
###

## Install
$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash ~/Miniconda3-latest-Linux-x86_64.sh

## Create env
$ cd data_science_for_life_science_summer_2026_project_python
$ conda env create -f environment.yml

## Activate
$ conda activate data_science_in_life_sciences_project_2026_group_1

###
# Data
###

## still in ~/workspace/data_science_for_life_science_summer_2026_project_python
$ source setup_environment.sh 

## Download raw archives

### GSE164378_RAW.tar
curl -o GSE164378_RAW.tar https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/GSE164378_RAW.tar

### GSE164378_sc.meta.data_3P.csv.gz
curl -o GSE164378_sc.meta.data_3P.csv.gz https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/GSE164378_sc.meta.data_3P.csv.gz

## Create and save MuData datasets
## -> all data is saved in /data/scratch/${USER}/.data_science_project

$ cd scripts
### 1. Extract full dataset
$ ./1_load_full_dataset.py

### 2. Optionally: create subsample 
$ ./2_create_subsample_datasets.py --subsample_size 10_000

### 3. Create test/training split
$ ./3_split_dataset.py --test_split_size 40 # i.e: test/training = 40%/60%, based on full dataset
$ ./3_split_dataset.py --test_split_size 40 --subsample_size 10_000 # based on the 10_000 subsample

### 4. Feature selection 
### - Safely reduce genes, based on a specific split
### - This saves a copy of the split data
$ ./4_feature_selection.py --test_split_size 40
```

No that we have all the data ready and we can submit the interesting jobs to the computing nodes. 

***Note:*** before we can run the MLP/IG scoring, we will have to train and save the model!

```bash
###
# Train the MLP model
###

# TODO: submit a batch job to the cluster
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
