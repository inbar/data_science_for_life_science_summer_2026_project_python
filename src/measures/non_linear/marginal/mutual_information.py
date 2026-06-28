import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif

import config


def compute_mi_scores(cell_type: str,
                      expression_levels_df: pd.DataFrame,
                      target_vector: pd.Series,
                      k_neighbors: int,
                      seed: int):
    score = mutual_info_classif(
        X=expression_levels_df,
        y=target_vector,
        n_neighbors=k_neighbors,
        random_state=seed
    )

    return cell_type, score


def calculate_scores(expression_levels_df: pd.DataFrame,
                     labeling_df: pd.DataFrame,
                     k_neighbors=3,
                     seed=config.SEED):
    """Computes the marginal KSG Mutual Information for every gene against
    every cell type.

    Parameters
    ----------
    expression_levels_df : pd.DataFrame
        The raw or normalized log-counts matrix. Rows are Cells, Columns are Genes.
    labeling_df: A pandas DataFrame of shape (n_cells, n_cell_types) containing the
        true cell-type labels.

        A pandas DataFrame of shape (n_cells, n_genes) containing the gene expression matrix.
    k_neighbors : int
        The number of nearest neighbors for the KSG estimator (typically 3 or 5).
    seed : int
        Random number generator seed

    Returns
    -------
    results: pd.DataFrame
        A long-form dataframe or a wide matrix mapping Genes to their MI scores.
      """

    # score_dict = {}
    # for cell_type in labeling_df.columns:
    #     target_vector = pd.Series(labeling_df[cell_type])
    #     scores = compute_mi_scores(
    #         expression_levels_df=expression_levels_df,
    #         target_vector=target_vector,
    #         k_neighbors=k_neighbors,
    #         seed=seed
    #     )
    #
    #     score_dict[cell_type] = scores

    test_cells = expression_levels_df.sample(
        n=min(200, len(expression_levels_df)), random_state=42).index

    test_expr = expression_levels_df.loc[test_cells].iloc[:, :50]
    test_labels = labeling_df.loc[test_cells].iloc[:, :3]

    # n_jobs=-1 will utilize all available CPU cores automatically
    n_jobs = -1

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_mi_scores)(
            cell_type,
            test_expr,
            test_labels[cell_type],
            k_neighbors,
            seed
        )
        for cell_type in test_labels.columns
    )

    score_dict = {cell_type: scores for cell_type, scores in results}
    genes = test_expr.columns
    results = pd.DataFrame(score_dict,
                           index=genes)

    return results