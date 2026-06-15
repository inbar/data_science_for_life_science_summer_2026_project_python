"""Project-wide configuration: paths, seeds, constants, plotting style.

Centralising these keeps the notebook free of magic numbers and makes the whole
pipeline reproducible from a single import.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the project root, i.e. the parent of ``src/``).
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR_PATH = ROOT / "data" / "raw"
PROCESSED_DATA_DIR_PATH = ROOT / "data" / "processed"
RESULTS_DIR_PATH = ROOT / "results"
FIGURES_DIR_PATH = RESULTS_DIR_PATH / "figures"
TABLES_DIR_PATH = RESULTS_DIR_PATH / "tables"

for path in (RAW_DATA_DIR_PATH,
             PROCESSED_DATA_DIR_PATH,
             RESULTS_DIR_PATH,
             FIGURES_DIR_PATH,
             TABLES_DIR_PATH):
    path.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 0

# --------------------------------------------------------------------------- #
# Dataset / preprocessing constants
# --------------------------------------------------------------------------- #

FTP_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/"

SUBSAMPLE_SIZE = 25_000          # stratified cell subsample size (pitch spec)
N_HVG = 2_000                 # highly variable genes -> shared feature space
N_PCS = 50                    # PCA components for UMAP / Harmony
CELLTYPE_LEVELS = {           # metadata column -> human label
    "celltype.l1": "L1 (8 types)",
    "celltype.l2": "L2 (~30 types)",
    "celltype.l3": "L3 (57 types)",
}
PRIMARY_LEVEL = "celltype.l2"  # primary benchmark granularity
DONOR_KEY = "donor"
MIN_DRIVERS = 2                # exclude cell types with |D_c| < MIN_DRIVERS

# --------------------------------------------------------------------------- #
# Method labels (consistent ordering / naming everywhere)
# --------------------------------------------------------------------------- #
METHODS = ["spearman", "partial_corr", "mi_ksg", "ig_mlp"]
METHOD_LABELS = {
    "spearman": "Spearman",
    "partial_corr": "Partial correlation",
    "mi_ksg": "Mutual information (KSG)",
    "ig_mlp": "Integrated Gradients (MLP)",
}
# 2x2 design coordinates: (linear?, marginal?)
METHOD_GRID = {
    "spearman":     ("linear",    "marginal"),
    "partial_corr": ("linear",    "conditional"),
    "mi_ksg":       ("nonlinear", "marginal"),
    "ig_mlp":       ("nonlinear", "conditional"),
}

# --------------------------------------------------------------------------- #
# Environment hint (the dedicated conda env created for this project)
# --------------------------------------------------------------------------- #
ENV_NAME = "data_science_in_life_sciences_project_2026_group_1"



def fig_path(name: str, ext: str = "pdf") -> Path:
    return FIGURES_DIR_PATH / f"{name}.{ext}"


def tab_path(name: str, ext: str = "csv") -> Path:
    return TABLES_DIR_PATH / f"{name}.{ext}"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
