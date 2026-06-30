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

    dataset = dataset_persistence.load_or_create_subsample(
        subsample_size=subsample_size,
        level=level)

    training_data, test_data = splitting.split(dataset,
                                               test_split_size=test_split_size / 100,
                                               seed=seed)

    # Scaling both datasets and saving as a layer to be used in downstream computations
    rna_preprocessing.scale_to_layer(training_data, test_data)

    splits_persistence.save_split(training_data=training_data,
                                  test_data=test_data,
                                  test_split_size=test_split_size,
                                  seed=seed,
                                  subsample_size=subsample_size,
                                  level=level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_size", type=int,
                        default=config.DEFAULT_SUBSAMPLE_SIZE)
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--test_split_size", type=int,
                        default=config.DEFAULE_TEST_SPLIT_SIZE)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parsed_args = parser.parse_args()

    main(parsed_args)
