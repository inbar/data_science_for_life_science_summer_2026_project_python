#!/usr/bin/env python
import argparse
import warnings

from anndata import ImplicitModificationWarning
from pandas.errors import PerformanceWarning

from scripts.helpers.args import bool_value, dump_args
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


def main(level: str = config.DEFAULT_LEVEL,
         force_recreate: bool = False):
    dataset_persistence.load_or_create_full_dataset(level=level,
                                                    force_recreate=force_recreate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--force_recreate", type=bool_value, default=False)
    parsed_args = parser.parse_args()

    log.info(f"Load full dataset")
    log.info("==================")
    dump_args(parsed_args, log)

    main(
        level=parsed_args.level,
        force_recreate=parsed_args.force_recreate
    )
