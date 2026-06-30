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

warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

import logging

logs.setup_logging(__file__)
log = logging.getLogger(__file__)


def main(args):
    subsample_size = args.subsample_size
    level = args.level
    seed = args.seed
    force_recreate = args.force_recreate

    dataset = dataset_persistence.load_or_create_subsample(
        subsample_size=subsample_size,
        level=level,
        seed=seed,
        force_recreate=force_recreate
    )

    dataset_persistence.save_subsampled_dataset(dataset=dataset,
                                                subsample_size=subsample_size,
                                                seed=seed,
                                                level=level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_size", type=int, default=config.DEFAULT_SUBSAMPLE_SIZE)
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--force_recreate", type=bool, default=False)
    parsed_args = parser.parse_args()

    main(parsed_args)
