#!/usr/bin/env python

# To run the scripts run:
# source setup_environment.sh
# In the project root

import argparse
import logging

from src import config
from src import logs
from src.deep_learning import tuning
from src.persistence import splits as split_persistence
from src.persistence import subsampling
from src.preprocessing import rna as rna_preprocessing
from src.preprocessing import splitting
from src.persistence import hyperparameters

logs.setup_logging(__file__)
log = logging.getLogger(__file__)


SUBSAMPLE_SIZE_FOR_TUNING=10_000
TEST_SPLIT_SIZE_FOR_TUNING=15

def main(args):
    level = args.level
    seed = args.seed
    test_split_size = args.test_split_size
    n_trials = args.n_trials

    log.info(f"Running Training")
    log.info("===================")
    for k, v in vars(parsed_args).items():
        log.info(f"   {k}: {v}")
    log.info("")

    # To avoid hyperparameter data leakage, we do the optimization explicitly
    # on a subsample of the training data and do not touch the test data.
    #
    # This makes sure that the parameters never saw the test cells and
    # avoids the obstacle that the hyperparameters are optimized or over sensitive
    # to any cell in the test dataset.
    training_data = split_persistence.load_training_data(
        split_name=split_persistence.HVG_SPLIT_NAME,
        test_split_size=test_split_size,
        seed=seed,
        level=level
    )

    # Subsample
    dataset = subsampling.subsample(dataset=training_data,
                                    level=level,
                                    subsample_size=SUBSAMPLE_SIZE_FOR_TUNING,
                                    seed=seed)

    # Split
    training_data, test_data = splitting.split(dataset,
                                               test_split_size=TEST_SPLIT_SIZE_FOR_TUNING,
                                               seed=seed)

    rna_dataset_training = training_data["rna"]
    target_df_training = rna_preprocessing.build_target_df(rna_dataset_training,
                                                           level)

    rna_dataset_test = test_data["rna"]
    target_df_test = rna_preprocessing.build_target_df(rna_dataset_test,
                                                       level)

    study = tuning.tune(
        training_data=rna_dataset_training,
        test_data=rna_dataset_test,
        labeling_df_training=target_df_training,
        labeling_df_test=target_df_test,
        n_trials=n_trials
    )

    hyperparameters.save_best_params(study.best_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=config.N_TRIALS)
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--test_split_size", type=int,
                        default=config.DEFAULE_TEST_SPLIT_SIZE)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parsed_args = parser.parse_args()

    main(parsed_args)
