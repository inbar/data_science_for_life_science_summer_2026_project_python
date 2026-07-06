#!/usr/bin/env python
"""Protein-derived ground truth D_c: for each cell type, the surface proteins
significantly elevated one-vs-rest on CLR-normalized ADT, mapped to their
encoding gene(s) (UniProt/HGNC curation, src.mappings) and intersected with the
scored gene universe. Independent of all four RNA scoring methods.

Depends on the subsampled dataset (for ADT) and the feature-selected split (for
the gene universe) -- runs any time after 5_feature_selection.py; does not
depend on model training or scoring.

# To run the scripts run:
# source setup_environment.sh
# In the project root
"""
import argparse
from pathlib import Path

from src import config
from src import ground_truth as ground_truth_builder
from src import logs
from src.persistence import datasets as dataset_persistence
from src.persistence import ground_truth as ground_truth_persistence
from src.persistence import splits as split_persistence
from src.preprocessing import adt as adt_preprocessing

import logging

logs.setup_logging(__file__)
log = logging.getLogger(__file__)


def main(args):
    level = args.level
    subsample_size = args.subsample_size
    test_split_size = args.test_split_size
    seed = args.seed
    root_dir = args.root_dir

    log.info(f"Ground truth")
    log.info("=============")
    for k, v in vars(parsed_args).items():
        log.info(f"   {k}: {v}")
    log.info("")

    dataset = dataset_persistence.load_or_create_subsample(
        subsample_size=subsample_size, level=level, seed=seed)
    adt_dataset = dataset["adt"]

    # The HVG split subsets train/test to the identical gene set, so either
    # side gives the same universe; test is loaded since it is smaller.
    test_data_rna = split_persistence.load_test_data(
        split_name=split_persistence.HVG_SPLIT_NAME,
        test_split_size=test_split_size, subsample_size=subsample_size,
        seed=seed, level=level)["rna"]
    genes_of_interest = test_data_rna.var["gene_name"].tolist()

    drivers, marker_proteins = ground_truth_builder.build_ground_truth(
        adt_dataset, genes_of_interest=genes_of_interest, level=level,
        layer=adt_preprocessing.LAYER_NAME_CENTERED_LOG_RATIO)

    for cell_type, driver_genes in drivers.items():
        log.info(f"  {cell_type}: {len(driver_genes)} drivers")

    ground_truth_persistence.save_ground_truth(
        drivers, root_dir=root_dir, level=level, subsample_size=subsample_size,
        test_split_size=test_split_size, seed=seed)

    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--subsample_size", type=int,
                        default=config.DEFAULT_SUBSAMPLE_SIZE)
    parser.add_argument("--test_split_size", type=int,
                        default=config.DEFAULE_TEST_SPLIT_SIZE)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--root_dir", type=str, default=str(config.LOCAL_DATA_ROOT),
                        help="Repo-local results root (matches the existing "
                             "local_data/ground_truth/... convention).")
    parsed_args = parser.parse_args()
    parsed_args.root_dir = Path(parsed_args.root_dir)

    main(parsed_args)
