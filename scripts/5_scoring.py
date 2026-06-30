#!/usr/bin/env python

# To run the scripts run:
# source setup_environment.sh
# In the project root

import argparse
import warnings

import pandas as pd
from anndata import ImplicitModificationWarning, AnnData
from pandas.errors import PerformanceWarning
from sklearn.preprocessing import StandardScaler

from src.deep_learning.gene_expression_mlp_model import GeneExpressionModel
from src.deep_learning import gene_expression_mlp_model
from src import config
from src.logs import get_logger
from src.persistence import splits as split_persistence
from src.preprocessing import rna as rna_preprocessing
from src.measures.scoring.linear.marginal import spearman_correlation
from src.measures.scoring.linear.conditional import \
    ledoit_wolf_partial_correlation
from src.measures.scoring.non_linear.marginal import mutual_information_ksg
from src.measures.scoring.non_linear.conditional import \
    mlp_with_integrated_gradient

warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

log = get_logger()


def run_spearman(rna_data: AnnData,
                 labeling_df: pd.DataFrame) -> pd.DataFrame:
    expression_levels_df = rna_data.to_df().copy()
    expression_levels_df.columns = rna_data.var["gene_name"]

    return spearman_correlation.calculate_scores(expression_levels_df,
                                                 labeling_df)


def run_partial_correlation(rna_data: AnnData,
                            labeling_df: pd.DataFrame,
                            fitted_scaler: StandardScaler) -> pd.DataFrame:
    scaled_expression_levels_df = pd.DataFrame(
        data=fitted_scaler.transform(rna_data.to_df()),
        index=rna_data.obs_names,
        columns=rna_data.var["gene_name"])

    return ledoit_wolf_partial_correlation.calculate_scores(
        expression_levels_df=scaled_expression_levels_df,
        labeling_df=labeling_df
    )


def run_mutual_information(rna_data: AnnData,
                           labeling_df: pd.DataFrame,
                           k_neighbors: int,
                           seed: int) -> pd.DataFrame:
    expression_levels_df = rna_data.to_df().copy()
    expression_levels_df.columns = rna_data.var["gene_name"]

    return mutual_information_ksg.calculate_scores(expression_levels_df,
                                                   labeling_df,
                                                   k_neighbors=k_neighbors,
                                                   seed=seed)


def run_mlp_ig(trained_model: GeneExpressionModel,
               rna_data: AnnData,
               labeling_df: pd.DataFrame,
               fitted_scaler: StandardScaler):
    scaled_expression_levels_df = pd.DataFrame(
        data=fitted_scaler.transform(rna_data.to_df()),
        index=rna_data.obs_names,
        columns=rna_data.var["gene_name"])

    return mlp_with_integrated_gradient.calculate_scores(trained_model,
                                                         expression_levels_df=scaled_expression_levels_df,
                                                         labeling_df=labeling_df)


def get_trained_model(training_data: AnnData,
                      test_split_size: int,
                      seed: int,
                      subsample_size: int):
    n_genes = training_data.n_vars
    n_cells = training_data.n_obs

    return gene_expression_mlp_model.load_trained_model(
        n_genes=n_genes,
        n_cells=n_cells,
        test_split_size=test_split_size,
        seed=seed,
        subsample_size=subsample_size
    )


def main(args):
    subsample_size = args.subsample_size
    level = args.level
    test_split_size = args.test_split_size
    seed = args.seed
    method = args.method
    k_neighbors = args.k_neighbors

    log.info(f"Running Scoring")
    log.info("==========================")
    for k, v in vars(parsed_args).items():
        log.info(f"   {k}: {v}")
    log.info("")

    log.info("Loading split data...")
    log.info("")
    training_data, test_data = split_persistence.load_split_data(
        split_name=split_persistence.HVG_SPLIT_NAME,
        test_split_size=test_split_size,
        subsample_size=subsample_size,
        seed=seed,
        level=level
    )

    rna_training_data = training_data["rna"]
    rna_test_data = test_data["rna"]
    target_df = rna_preprocessing.build_target_df(rna_test_data, level)

    log.info("Training data (RNA modality):")
    log.info("-----------------------------")
    log.info(f"  n_cells (rows): {rna_training_data.n_obs}")
    log.info(f"  n_genes (cols): {rna_training_data.n_vars}")
    log.info("")

    log.info("Test data (RNA modality):")
    log.info("-----------------------------")
    log.info(f"  n_cells (rows): {rna_test_data.n_obs}")
    log.info(f"  n_genes (cols): {rna_test_data.n_vars}")
    log.info("")

    scaler = StandardScaler()
    scaler.fit(training_data["rna"].to_df())

    match method:
        case m if m == config.METHOD_SPEARMAN:
            results = run_spearman(rna_test_data, target_df)
        case m if m == config.METHOD_PC:
            results = run_partial_correlation(rna_test_data,
                                              target_df,
                                              scaler)
        case m if m == config.METHOD_MI:
            results = run_mutual_information(rna_test_data,
                                             target_df,
                                             seed=seed,
                                             k_neighbors=k_neighbors)
        case m if m == config.METHOD_MLP:

            trained_model = get_trained_model(training_data=rna_training_data,
                                              test_split_size=test_split_size,
                                              seed=seed,
                                              subsample_size=subsample_size)

            results = run_mlp_ig(rna_test_data,
                                 target_df,
                                 scaler)
        case _:
            raise ValueError(f"No such method: {method}")

    log.info("Done.")

    print(results)
    # TODO: persist results


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
    parser.add_argument("--k_neighbors", type=int,
                        default=config.DEFAULT_K_NEIGHBORS,
                        help="Only relevant for Mutual Information. Ignored otherwise.")
    parsed_args = parser.parse_args()

    main(parsed_args)
