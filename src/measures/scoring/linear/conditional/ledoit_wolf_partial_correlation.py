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

    log.info("Computing Linear/Conditional scoring: partial correlation (Leodit-Wolf)")
    num_genes = expression_levels_df.columns.size

    # Horizontally stack X and Y into a single matrix for global covariance.
    # Shape: (n_cells, n_genes + n_cell_types)
    combined_matrix = np.hstack([expression_levels_df.values, labeling_df.values])

    # Compute the Ledoit-Wolf shrunk covariance matrix.
    # sklearn expects observations as rows, features as columns with shape (n_samples, n_features)
    # The output matrix is then of shape (n_features, n_features),
    # i.e: all columns vs. all columns (of the input)
    shrunk_cov_matrix, _ = ledoit_wolf(combined_matrix)

    # Pick the values of interest: rectangle: X columns vs Y columns.
    # This extracts the rows corresponding to genes, and columns corresponding to targets.
    association_slice = shrunk_cov_matrix[:num_genes, num_genes:]

    results = pd.DataFrame(
        association_slice,
        index=expression_levels_df.columns,
        columns=labeling_df.columns
    )

    return results
