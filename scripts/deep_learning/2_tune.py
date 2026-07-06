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
from src.persistence import parameter_tuning

logs.setup_logging(__file__)
log = logging.getLogger(__file__)


SUBSAMPLE_SIZE_FOR_TUNING=10_000
TEST_SPLIT_SIZE_FOR_TUNING=15

def main(args):
    level = args.level
    seed = args.seed
    test_split_size = args.test_split_size
    n_trials = args.n_trials
    subsample_size = args.subsample_size

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
    #
    # NOTE: subsample_size defaults to -1 (the full dataset's training split) so
    # tuning draws from the largest, most representative pool regardless of what
    # subsample_size a downstream training/scoring run uses; pass a matching
    # --subsample_size explicitly if that split hasn't been prepared (e.g. in a
    # small/dev run) and you want to tune from the smaller split instead.
    training_data = split_persistence.load_training_data(
        split_name=split_persistence.HVG_SPLIT_NAME,
        test_split_size=test_split_size,
        subsample_size=subsample_size,
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
                                               level=level,
                                               test_split_size=TEST_SPLIT_SIZE_FOR_TUNING,
                                               seed=seed)

    rna_dataset_training = training_data["rna"]
    target_df_training = rna_preprocessing.build_target_df(rna_dataset_training,
                                                           level)

    rna_dataset_test = test_data["rna"]
    target_df_test = rna_preprocessing.build_target_df(rna_dataset_test,
                                                       level)

    # training.train() (called per-trial by tuning.objective) reads the "scaled"
    # layer -- it must be computed here or every trial fails with a KeyError.
    rna_preprocessing.apply_scaling_to_split_data(rna_dataset_training,
                                                  rna_dataset_test)

    study = tuning.tune(
        training_data=rna_dataset_training,
        test_data=rna_dataset_test,
        labeling_df_training=target_df_training,
        labeling_df_test=target_df_test,
        n_trials=n_trials
    )

    # NOTE: root_dir must match where 1_train.py's load_best_params() looks
    # (config.LOCAL_DATA_ROOT) -- the module's own default (config.PERSISTENCE_DIR,
    # outside the repo) would silently save somewhere 1_train.py never checks.
    parameter_tuning.save_best_params(study.best_params,
                                      root_dir=config.LOCAL_DATA_ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=config.N_TRIALS)
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--test_split_size", type=int,
                        default=config.DEFAULE_TEST_SPLIT_SIZE)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--subsample_size", type=int,
                        default=config.DEFAULT_SUBSAMPLE_SIZE,
                        help="Which HVG split's training data to draw the tuning "
                             "subsample from (default -1 = full dataset).")
    parsed_args = parser.parse_args()

    main(parsed_args)
