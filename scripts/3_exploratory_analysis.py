#!/usr/bin/env python
"""QC, UMAP (+/- Harmony batch correction), and ADT label-validation artifacts.

Runs on the subsampled dataset -- right after subsampling and before splitting,
since these diagnostics (is QC OK, is there a donor batch effect, do the WNN
labels agree with the independent protein channel) inform the rest of the run
rather than depending on it. The HVGs used here are a throwaway, exploratory-only
selection for the embedding; the "official" gene universe used for scoring is
selected later (5_feature_selection.py), on the training split only, to avoid
leakage.

# To run the scripts run:
# source setup_environment.sh
# In the project root
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OpenMP runtime clash
                                                        # (Harmony pulls in a second
                                                        # OpenMP runtime on Windows)

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import ImplicitModificationWarning
from pandas.errors import PerformanceWarning

from src import config
from src import logs
from src.exploratory_analysis import dim_reduction
from src.persistence import datasets as dataset_persistence
from src.preprocessing import adt as adt_preprocessing
from src.preprocessing import rna as rna_preprocessing

warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

import logging

logs.setup_logging(__file__)
log = logging.getLogger(__file__)

# Canonical surface-protein markers used to sanity-check the annotated lineages.
ADT_VALIDATION_PROTEINS = ["CD3-1", "CD4-1", "CD8", "CD19", "CD20", "CD14", "CD16",
                           "CD56-1", "CD11c", "HLA-DR", "CD123", "CD34"]

N_EXPLORATORY_HVGS = 2_000


def _dense(x):
    return np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)


def get_output_dir(subsample_size, level, seed, root_dir):
    # Depends only on (level, subsample_size, seed) -- there is no train/test
    # split yet at this stage, so this deliberately does not use
    # persistence.path_tools's split-aware naming.
    subdir = dataset_persistence.SUBSAMPLE_DATASET_SUBDIR_TEMPLATE.format(
        subsample_size=subsample_size, seed=seed)
    output_dir = root_dir / "exploratory" / level / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(args):
    level = args.level
    subsample_size = args.subsample_size
    seed = args.seed
    root_dir = args.root_dir

    log.info(f"Exploratory analysis")
    log.info("=====================")
    for k, v in vars(parsed_args).items():
        log.info(f"   {k}: {v}")
    log.info("")

    output_dir = get_output_dir(subsample_size, level, seed, root_dir)

    dataset = dataset_persistence.load_or_create_subsample(
        subsample_size=subsample_size, level=level, seed=seed)
    rna_dataset, adt_dataset = dataset["rna"].copy(), dataset["adt"].copy()
    log.info(f"{rna_dataset.n_obs} cells | RNA {rna_dataset.n_vars} genes | "
             f"ADT {adt_dataset.n_vars} proteins")

    # --- QC ---
    qc = rna_dataset.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].rename(
        columns={"n_genes_by_counts": "genes / cell", "total_counts": "UMI / cell",
                 "pct_counts_mt": "% mitochondrial"})
    qc.to_csv(output_dir / "qc.csv", index=False)

    # --- UMAP (+/- Harmony), on an exploratory-only HVG selection ---
    rna_preprocessing.annotate_highly_variable_genes(rna_dataset, n_top=N_EXPLORATORY_HVGS)
    hvg = rna_dataset.var_names[rna_dataset.var["highly_variable"]]
    rna_dataset_hvg = rna_dataset[:, hvg].copy()
    sc.pp.scale(rna_dataset_hvg, max_value=10)
    dim_reduction.perform_pca_in_place(rna_dataset_hvg)
    dim_reduction.perform_umap_in_place(rna_dataset_hvg)
    dim_reduction.perform_pca_harmony_in_place(rna_dataset_hvg)
    dim_reduction.perform_umap_harmony_in_place(rna_dataset_hvg)
    np.savez(output_dir / "embeddings.npz",
             umap=rna_dataset_hvg.obsm[dim_reduction.OBSM_NAME_UMAP],
             umap_harmony=rna_dataset_hvg.obsm[dim_reduction.OBSM_NAME_UMAP_HARMONY],
             celltype=rna_dataset.obs[level].astype(str).values,
             donor=rna_dataset.obs[config.DONOR_KEY].astype(str).values)
    log.info(f"Embeddings saved (HVGs used for exploration only: {len(hvg)})")

    # --- ADT label validation: mean CLR protein level per cell type ---
    validation_proteins = [p for p in ADT_VALIDATION_PROTEINS
                           if p in adt_dataset.var_names]
    clr_df = pd.DataFrame(
        _dense(adt_dataset.layers[adt_preprocessing.LAYER_NAME_CENTERED_LOG_RATIO]),
        columns=adt_dataset.var_names)
    clr_df["celltype"] = rna_dataset.obs[level].astype(str).values
    means = clr_df.groupby("celltype")[validation_proteins].mean()
    ((means - means.mean()) / means.std()).to_csv(output_dir / "adt_validation.csv")

    log.info(f"Done. Artifacts written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--subsample_size", type=int,
                        default=config.DEFAULT_SUBSAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--root_dir", type=str, default=str(config.LOCAL_DATA_ROOT),
                        help="Repo-local results root (exploratory artifacts are "
                             "meant to be inspected from the repo, unlike the "
                             "large cached datasets under PROCESSED_DATA).")
    parsed_args = parser.parse_args()
    parsed_args.root_dir = Path(parsed_args.root_dir)

    main(parsed_args)
