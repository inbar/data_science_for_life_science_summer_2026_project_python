from __future__ import annotations

from pathlib import Path

# Paths
PROJECT_REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_HOME = Path.home() / ".data_science_project"

# Persistence
PERSISTENCE_DIR = PROJECT_HOME / "persistence"
DATA_DIR = PROJECT_HOME / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PROCESSED_DATA = DATA_DIR / "processed_data"

RESULTS_DIR_PATH = PERSISTENCE_DIR / "results"
FIGURES_DIR_PATH = RESULTS_DIR_PATH / "figures"
TABLES_DIR_PATH = RESULTS_DIR_PATH / "tables"

LOGS_DIR_PATH = PROJECT_HOME / "logs"

for path in (RESULTS_DIR_PATH,
             FIGURES_DIR_PATH,
             TABLES_DIR_PATH):
    path.mkdir(parents=True, exist_ok=True)





FTP_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/"

# Defaults

DEFAULT_SEED = 42
DEFAULT_SUBSAMPLE_SIZE = -1  # stratified cell subsample size (pitch spec)
N_TOP_HVGs = 2_000  # highly variable genes -> shared feature space
N_PCS = 50  # PCA components for UMAP / Harmony
CELLTYPE_LEVELS = {  # metadata column -> human label
    "celltype.l1": "L1 (8 types)",
    "celltype.l2": "L2 (~30 types)",
    "celltype.l3": "L3 (57 types)",
}
DEFAULT_LEVEL = "celltype.l2"  # primary benchmark granularity
DONOR_KEY = "donor"
MIN_DRIVERS = 2  # exclude cell types with |D_c| < MIN_DRIVERS
DEFAULE_TEST_SPLIT_SIZE = 15
DEFAULT_K_NEIGHBORS = 3
DEFAULT_N_EPOCHS = 3

# Method labels

METHOD_SPEARMAN = "spearman"
METHOD_PC = "partial_corr"
METHOD_MI = "mi_ksg"
METHOD_MLP = "ig_mlp"

METHODS = [METHOD_SPEARMAN, METHOD_PC, METHOD_MI, METHOD_MLP]
METHOD_LABELS = {
    "spearman": "Spearman",
    "partial_corr": "Partial correlation",
    "mi_ksg": "Mutual information (KSG)",
    "ig_mlp": "Integrated Gradients (MLP)",
}

METHOD_GRID = {
    "spearman": ("linear", "marginal"),
    "partial_corr": ("linear", "conditional"),
    "mi_ksg": ("nonlinear", "marginal"),
    "ig_mlp": ("nonlinear", "conditional"),
}


# Conda env
ENV_NAME = "data_science_in_life_sciences_project_2026_group_1"
