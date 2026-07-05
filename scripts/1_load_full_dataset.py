#!/usr/bin/env python
import argparse
import warnings

from anndata import ImplicitModificationWarning
from pandas.errors import PerformanceWarning

from scripts.helpers.arg_types import bool_value
from src import config
from src import logs
from src.persistence import datasets as dataset_persistence

# To run the scripts run:
# source setup_environment.sh
# In the project root


warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

import logging

logs.setup_logging(__file__)
log = logging.getLogger(__file__)


def main(args):
    level = args.level
    force_recreate = args.force_recreate

    log.info(f"Load full dataset")
    log.info("==================")
    for k, v in vars(parsed_args).items():
        log.info(f"   {k}: {v}")
    log.info("")

    dataset_persistence.load_or_create_full_dataset(level=level,
                                                    force_recreate=force_recreate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--force_recreate", type=bool_value, default=False)
    parsed_args = parser.parse_args()

    main(parsed_args)
