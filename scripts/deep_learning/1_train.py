#!/usr/bin/env python

# To run the scripts run:
# source setup_environment.sh
# In the project root

import argparse
import warnings

from anndata import ImplicitModificationWarning
from pandas.errors import PerformanceWarning

from src.deep_learning.gene_expression_mlp_model import GeneExpressionModel
from src import config
from src import logs
from src.deep_learning import training
from src.persistence import parameter_tuning as hyperparam_persistence
from src.persistence import models as model_persistence
from src.persistence import splits as split_persistence
from src.preprocessing import rna as rna_preprocessing

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
    n_epochs = args.n_epochs
    tag = args.tag

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

    log.info("Loading hyperparameter values from file...")
    hyperparams = hyperparam_persistence.load_best_params(
        root_dir=config.LOCAL_DATA_ROOT)

    n_genes = training_data_rna.n_vars
    n_celltypes = training_data_rna.obs[level].nunique()

    log.info("Creating new model instance...")
    model = GeneExpressionModel(
        input_dim=n_genes,
        output_dim=n_celltypes,
        dropout_rate=hyperparams["input_dropout"]
    )

    trained_model, _history = training.train(model=model,
                                             training_data=training_data_rna,
                                             labeling_df=target_df,
                                             n_epochs=n_epochs,
                                             learning_rate=hyperparams["learning_rate"],
                                             weight_decay=hyperparams["weight_decay"],
                                             batch_size=hyperparams["batch_size"])

    model_persistence.save_trained_model_weights(trained_model,
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
    parser.add_argument("--n_epochs", type=int,
                        default=config.DEFAULT_N_EPOCHS)
    parser.add_argument("--tag", type=str, default=config.DEFAULT_TAG)
    parsed_args = parser.parse_args()

    main(parsed_args)
