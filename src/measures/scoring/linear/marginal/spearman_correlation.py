import numpy as np
import pandas as pd
from scipy.stats import rankdata

import logging

log = logging.getLogger(__file__)


def calculate_scores(expression_levels_df: pd.DataFrame,
                     labeling_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Spearman correlation scores between a gene data matrix
    and target cell type vectors.

    Loops over all columns in the target matrix (cell types) and computes
    the correlation against the each column in the data matrix.

    The columns of the data matrix are vectors of expression levels for each
    gene across all cells: columns are genes and values are expression data
    for each cell (row).

    The columns of the target matrix are vectors of cell-type labeling data
    for each cell. The labeling is on-hot encoded: columns are cell types,
    and values are 1 if the cell (row) belongs to that type, and 0 otherwise.

    Parameters
    ----------
        expression_levels_df: A pandas DataFrame of shape (n_cells, n_genes)
            containing the gene expression matrix.
        labeling_df: A pandas DataFrame of shape (n_cells, n_cell_types) containing the
            true cell-type labels.

    Returns
    -------
    results: pd.DataFrame
        A DataFrame containing correlation scores between each gene and cell type.
    """
    log.info("Computing Linear/Marginal scoring: Spearman correlation")

    # Spearman rho == Pearson correlation on ranks. Rank each gene (column) across
    # cells once, then correlate against the (ranked) one-vs-rest indicator per cell
    # type. Ranking internally makes this a genuine Spearman regardless of whether
    # the caller already passed the rank-transformed layer.
    ranked_expression = np.apply_along_axis(rankdata, 0,
                                            expression_levels_df.values)
    centered = ranked_expression - ranked_expression.mean(axis=0, keepdims=True)
    gene_norms = np.sqrt((centered ** 2).sum(axis=0))
    gene_norms[gene_norms == 0] = np.nan

    columns = []
    for cell_type in labeling_df.columns:
        target_ranks = rankdata(labeling_df[cell_type].values.astype(float))
        target_centered = target_ranks - target_ranks.mean()
        target_norm = np.sqrt((target_centered ** 2).sum())
        columns.append((centered.T @ target_centered) / (gene_norms * target_norm))

    result_matrix = np.column_stack(columns)

    results = pd.DataFrame(data=result_matrix,
                           columns=labeling_df.columns,
                           index=expression_levels_df.columns)

    return results
