#!/usr/bin/env python

# To run the scripts run:
# source setup_environment.sh
# In the project root

import argparse

from src.persistence import datasets as dataset_persistence

from src import config
from src.logs import get_logger
import warnings
from pandas.errors import PerformanceWarning
from anndata import ImplicitModificationWarning

warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

log = get_logger()


def main(args):
    subsample_size = args.subsample_size
    level = args.level
    seed = args.seed

    dataset = dataset_persistence.load_or_create_subsample(
        subsample_size=subsample_size,
        level=level,
        seed=seed)

    dataset_persistence.save_subsampled_dataset(dataset=dataset,
                                                subsample_size=subsample_size,
                                                seed=seed,
                                                level=level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_size", type=int, default=config.DEFAULT_SUBSAMPLE_SIZE)
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parsed_args = parser.parse_args()

    main(parsed_args)
