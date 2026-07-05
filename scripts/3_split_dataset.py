#!/usr/bin/env python

# To run the scripts run:
# source setup_environment.sh
# In the project root

import argparse
import warnings

from anndata import ImplicitModificationWarning
from pandas.errors import PerformanceWarning

from src import config
from src import logs
from src.persistence import datasets as dataset_persistence
from src.persistence import splits as splits_persistence
from src.preprocessing import rna as rna_preprocessing
from src.preprocessing import splitting

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
    tag = args.tag

    log.info(f"Split subsample")
    log.info("=================")
    for k, v in vars(parsed_args).items():
        log.info(f"   {k}: {v}")
    log.info("")

    dataset = dataset_persistence.load_or_create_subsample(
        subsample_size=subsample_size,
        level=level)

    data_training, data_test = splitting.split(dataset,
                                               test_split_size=test_split_size,
                                               seed=seed, )

    rna_dataset_training = data_training["rna"]
    adt_dataset_training = data_training["adt"]

    rna_dataset_test = data_test["rna"]
    adt_dataset_test = data_test["adt"]

    log.info("Filtering...")
    rna_dataset_filtered_training, rna_dataset_filtered_test = rna_preprocessing.apply_basic_filtering_to_split_data(
        rna_dataset_training,
        rna_dataset_test,
        level)

    remaining_cells_training = rna_dataset_filtered_training.obs_names

    # Filter the cells in the adt modality accordingly
    # Note: we only filter out cells in the training dataset because we don't
    # care about uninteresting cells in training.
    cells_to_keep_training = adt_dataset_training.obs_names.isin(
        remaining_cells_training
    )

    adt_dataset_filtered_training = adt_dataset_training[
        cells_to_keep_training, :
    ]

    filtered_dataset_training = dataset_persistence.create_mudata_dataset_from_anndata(
        rna_dataset=rna_dataset_filtered_training,
        adt_dataset=adt_dataset_filtered_training
    )

    filtered_dataset_test = dataset_persistence.create_mudata_dataset_from_anndata(
        rna_dataset=rna_dataset_filtered_test,
        adt_dataset=adt_dataset_test
    )

    splits_persistence.save_split(training_data=filtered_dataset_training,
                                  test_data=filtered_dataset_test,
                                  test_split_size=test_split_size,
                                  seed=seed,
                                  subsample_size=subsample_size,
                                  level=level,
                                  tag=tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_size", type=int,
                        default=config.DEFAULT_SUBSAMPLE_SIZE)
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--test_split_size", type=int,
                        default=config.DEFAULE_TEST_SPLIT_SIZE)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--tag", type=str, default=config.DEFAULT_TAG)
    parsed_args = parser.parse_args()

    main(parsed_args)
