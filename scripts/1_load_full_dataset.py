#!/usr/bin/env python
import argparse
import warnings

from anndata import ImplicitModificationWarning, AnnData
from mudata import MuData
from pandas.errors import PerformanceWarning

from src import config
from src import logs
from src.persistence import datasets as dataset_persistence
from src.preprocessing import adt as adt_preprocessing
from src.preprocessing import normalization
from src.preprocessing import rna as rna_preprocessing

# To run the scripts run:
# source setup_environment.sh
# In the project root


warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

import logging

logs.setup_logging(__file__)
log = logging.getLogger(__file__)


def preprocess_and_filter(rna_dataset: AnnData, adt_dataset: AnnData,
                          level: str) -> MuData:
    """QC + normalize + basic-filter a raw (RNA, ADT) pair into the filtered
    MuData persisted as the "full dataset". Shared by this script's ``main()``
    (raw GEO source) and any alternate loader that sources the same two
    modalities from elsewhere (e.g. an already-processed MuData), so both paths
    apply identical preprocessing.
    """
    # Normalize. RNA: library-size normalize + log1p. ADT: centered-log-ratio
    # (CLR) -- the CITE-seq/muon standard, NOT the generic RNA-style normalize;
    # using the wrong one here silently produces an ADT dataset with no "clr"
    # layer, which breaks the CLR-based ground truth downstream.
    rna_preprocessing.calculate_qc_metrics_in_place(rna_dataset)
    normalization.normalize_in_place(rna_dataset)
    adt_preprocessing.normalize_in_place(adt_dataset)

    # Basic cell filtering
    log.debug(f"Before filtering: dataset.n_obs = {rna_dataset.n_obs} (cells)")
    log.debug(f"Before filtering: dataset.n_var = {rna_dataset.n_vars} (genes)")

    rna_dataset_filtered = rna_preprocessing.apply_basic_filtering(rna_dataset, level)

    log.debug(f"After filtering: dataset.n_obs = {rna_dataset_filtered.n_obs} (cells)")
    log.debug(f"After filtering: dataset.n_var = {rna_dataset_filtered.n_vars} (genes)")

    adt_dataset_filtered = adt_dataset[rna_dataset_filtered.obs_names, :].copy()

    return dataset_persistence.create_mudata_dataset_from_anndata(
        rna_dataset_filtered,
        adt_dataset_filtered
    )


def main(args):
    level = args.level
    force_recreate = args.force_recreate

    log.info(f"Load full dataset")
    log.info("==================")
    for k, v in vars(parsed_args).items():
        log.info(f"   {k}: {v}")
    log.info("")

    dataset = dataset_persistence.load_or_create_full_dataset(
        force_recreate=force_recreate, level=level)
    rna_dataset, adt_dataset = dataset["rna"], dataset["adt"]

    filtered_dataset = preprocess_and_filter(rna_dataset, adt_dataset, level)

    dataset_persistence.save_full_dataset(filtered_dataset, level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--force_recreate", type=bool, default=False)
    parsed_args = parser.parse_args()

    main(parsed_args)
