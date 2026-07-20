# Run jobs on the cluster

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

## Install conda
$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash ~/Miniconda3-latest-Linux-x86_64.sh

## Create conda env
$ cd data_science_for_life_science_summer_2026_project_python
$ conda env create -f environment.yml

## Activate conda env
$ conda activate data_science_in_life_sciences_project_2026_group_1

###
# Data
###

## still in ~/workspace/data_science_for_life_science_summer_2026_project_python
$ source setup_cluster_environment.sh 

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

### 3. Exploratory analysis (QC, UMAP +/- Harmony, ADT label validation)
### - Runs on the subsampled (or full) dataset, before splitting
$ ./3_exploratory_analysis.py --subsample_size 10_000

### 4. Create test/training split
$ ./4_split_dataset.py --test_split_size 40 # i.e: test/training = 40%/60%, based on full dataset
$ ./4_split_dataset.py --test_split_size 40 --subsample_size 10_000 # based on the 10_000 subsample

### 5. Feature selection 
### - Safely reduce genes, based on a specific split
### - This saves a copy of the split data
$ ./5_feature_selection.py --test_split_size 40

### 6. Ground truth
### - Protein-derived driver set D_c, built from CLR-normalized ADT + the gene
###   universe from step 5
$ ./6_ground_truth.py --test_split_size 40
```

Now that we have all the data ready and we can submit the interesting jobs to the computing nodes. 

***Note:*** before we can run the MLP/IG scoring, we will have to tune and train the model!

```bash
# Tune the model (with optional parameters)
$ TEST_SPLIT_SIZE=<size> N_TRIALS=<N> sbatch 102_tune_mlp_model.sh

# Train and save the model (with optional parameters)
$ sbatch 101_train_mlp_model.sh

# Run scoring
# Spearman
$ METHOD=spearman sbatch run_scoring.sh

# Partial correlation
$ METHOD=partial_corr sbatch run_scoring.sh

# Mutual Information
$ METHOD=mi_ksg,K_NEIGHBORS=5,TAG=KNN_5 sbatch --cpus-per-task=8 --time=06:00:00 run_scoring.sh

# IG/MLP
$ METHOD=ig_mlp sbatch run_scoring.sh
```


