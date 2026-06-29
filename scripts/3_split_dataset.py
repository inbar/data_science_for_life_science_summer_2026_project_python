#!/usr/bin/env python

# To run the scripts run:
# source setup_environment.sh
# In the project root

import argparse

from src.config import DEFAULE_TEST_SPLIT_SIZE, DEFAULT_SEED
from src.persistence import datasets as dataset_persistence
from src.persistence import splits as splits_persistence
from src.preprocessing import splitting

from src.logs import get_logger
from src.config import DEFAULT_LEVEL
import warnings
from pandas.errors import PerformanceWarning
from anndata import ImplicitModificationWarning

warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

log = get_logger()


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

    splits_persistence.save_split(training_data=training_data,
                                  test_data=test_data,
                                  test_split_size=test_split_size,
                                  seed=seed,
                                  subsample_size=subsample_size,
                                  level=level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_size", type=int, default=None)
    parser.add_argument("--level", type=str, default=DEFAULT_LEVEL)
    parser.add_argument("--test_split_size", type=str,
                        default=DEFAULE_TEST_SPLIT_SIZE)
    parser.add_argument("--seed", type=str, default=DEFAULT_SEED)
    parsed_args = parser.parse_args()

    main(parsed_args)
