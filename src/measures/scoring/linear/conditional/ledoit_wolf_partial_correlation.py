import numpy as np
import pandas as pd

from sklearn.covariance import ledoit_wolf

import logging

log = logging.getLogger(__file__)

def calculate_scores(expression_levels_df: pd.DataFrame,
                     labeling_df: pd.DataFrame) -> pd.DataFrame:
    """Compute covariance of each gene given the presence of
    all other genes in the matrix, (thus capturing conditional dependencies)

    Computes the Ledoit-Wolf shrunk covariance between features and targets.


    Parameters
    ----------
    expression_levels_df: A pandas DataFrame of shape (n_cells, n_genes)
        containing the gene expression matrix.
    labeling_df: A pandas DataFrame of shape (n_cells, n_cell_types)
        containing the true cell-type labels.

    Returns
    -------
    resutls: pd.DataFrame
        A DataFrame of shape (n_genes, n_cell_types) containing the
        shrunk covariance scores between each gene and cell type.
    """

    log.info("Computing Linear/Conditional scoring: partial correlation (Ledoit-Wolf)")
    num_genes = expression_levels_df.columns.size

    # Horizontally stack X and Y into a single matrix for a global covariance.
    # Shape: (n_cells, n_genes + n_cell_types)
    combined_matrix = np.hstack([expression_levels_df.values, labeling_df.values])

    # Ledoit-Wolf shrunk covariance -> guaranteed positive-definite and
    # well-conditioned (the empirical gene covariance is ill-conditioned: dropout,
    # collinearity, p ~ n).
    shrunk_cov_matrix, _ = ledoit_wolf(combined_matrix)

    # PARTIAL correlation requires the PRECISION matrix (inverse covariance): the
    # covariance block alone is only a *marginal* association. Invert, then convert
    # the precision to partial correlations:
    #     rho_partial(i, j) = -P_ij / sqrt(P_ii * P_jj)
    # which is the correlation of i and j conditioned on all other variables.
    precision_matrix = np.linalg.inv(shrunk_cov_matrix)
    d = np.sqrt(np.diag(precision_matrix))
    partial_correlation_matrix = -precision_matrix / np.outer(d, d)

    # Rectangle: gene rows vs. cell-type-target columns.
    association_slice = partial_correlation_matrix[:num_genes, num_genes:]

    results = pd.DataFrame(
        association_slice,
        index=expression_levels_df.columns,
        columns=labeling_df.columns
    )

    return results
