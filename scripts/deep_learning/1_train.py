#!/usr/bin/env python

# To run the scripts run:
# source setup_environment.sh
# In the project root

import argparse
import warnings

from anndata import ImplicitModificationWarning
from pandas.errors import PerformanceWarning

from src.deep_learning import training
from src import config
from src.logs import get_logger
from src.persistence import splits as split_persistence
from src.persistence import models as model_persistence
from src.preprocessing import rna as rna_preprocessing

warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

log = get_logger()

def main(args):
    subsample_size = args.subsample_size
    level = args.level
    test_split_size = args.test_split_size
    seed = args.seed
    n_epochs = args.n_epochs

    log.info(f"Running Training")
    log.info("===================")
    for k, v in vars(parsed_args).items():
        log.info(f"   {k}: {v}")
    log.info("")

    log.info("Loading split data...")
    log.info("")
    training_data = split_persistence.load_training_data(
        split_name=split_persistence.HVG_SPLIT_NAME,
        test_split_size=test_split_size,
        subsample_size=subsample_size,
        seed=seed,
        level=level
    )
    training_data_rna = training_data["rna"]
    target_df = rna_preprocessing.build_target_df(training_data_rna, level)

    trained_model = training.train(training_data_rna,
                                   target_df,
                                   n_epochs=n_epochs,
                                   level=level)

    model_persistence.save_trained_model_weights(trained_model,
                                      test_split_size=test_split_size,
                                      seed=seed,
                                      subsample_size=subsample_size)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=config.METHODS, type=str,
                        default=None)
    parser.add_argument("--subsample_size", type=int,
                        default=config.DEFAULT_SUBSAMPLE_SIZE)
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--test_split_size", type=int,
                        default=config.DEFAULE_TEST_SPLIT_SIZE)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--n_epochs", type=int,
                        default=config.DEFAULT_N_EPOCHS)
    parsed_args = parser.parse_args()

    main(parsed_args)
