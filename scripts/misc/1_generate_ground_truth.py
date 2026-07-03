#!/usr/bin/env python

# To run the scripts run:
# source setup_environment.sh
# In the project root

import argparse
import warnings

from anndata import ImplicitModificationWarning
from pandas.errors import PerformanceWarning

from src import ground_truth
from src import config
from src import logs

from src.persistence import splits as splits_persistence
from src.persistence import ground_truth as ground_truth_persistence

warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

import logging

logs.setup_logging(__file__)
log = logging.getLogger(__file__)


def main(args):
    subsample_size = args.subsample_size
    level = args.level
    test_split_size = args.test_split_size
    seed = args.seed

    log.info(f"Create Subsample")
    log.info("==================")
    for k, v in vars(parsed_args).items():
        log.info(f"   {k}: {v}")
    log.info("")

    test_data = splits_persistence.load_test_data(
        split_name=splits_persistence.HVG_SPLIT_NAME,
        test_split_size=test_split_size,
        subsample_size=subsample_size,
        seed=seed,
        level=level
    )

    test_data_adt = test_data.mod["adt"]
    test_data_rna = test_data.mod["rna"]

    genes_of_interest = test_data_rna.var["gene_name"].to_list()

    driver_gene_ground_truth = ground_truth.build_ground_truth(
        adt_dataset=test_data_adt,
        genes_of_interest=genes_of_interest)

    ground_truth_persistence.save_ground_truth(
        driver_gene_ground_truth,
        test_split_size=test_split_size,
        subsample_size=subsample_size,
        seed=seed,
        level=level
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_size", type=int, default=config.DEFAULT_SUBSAMPLE_SIZE)
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--test_split_size", type=int,
                        default=config.DEFAULE_TEST_SPLIT_SIZE)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parsed_args = parser.parse_args()

    main(parsed_args)