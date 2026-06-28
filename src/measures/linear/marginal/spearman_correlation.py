import numpy as np
import pandas as pd
import scipy.stats


def calculate_scores(expression_levels_df: pd.DataFrame,
                     labeling_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Spearman correlation scores between a gene data matrix and target cell type vectors.

    Loops over all columns in the target matrix (cell types) and computes
    the correlation against the each column in the data matrix.

    The columns of the data matrix are vectors of expression level for each
    gene: columns are genes and values are rank-transformed expression data
    for each cell (row).

    The columns of the target matrix are vectors of cell-type labeling data
    for each cell. The labeling is on-hot encoded: columns are cell types,
    and values are 1 if the cell (row) belongs to that type, and 0 otherwise.

    Parameters
    ----------
        expression_levels_df: A pandas DataFrame of shape (n_cells, n_genes)
            containing the *rank-transformed* gene expression matrix.
        labeling_df: A pandas DataFrame of shape (n_cells, n_cell_types) containing the
            true cell-type labels.

    Returns
    -------
    results: pd.DataFrame
        A DataFrame containing correlation scores between each
        gene and cell type.
        """
    results = []
    for cell_type in labeling_df.columns:
        # TODO: labeling_df[cell_type].values ?
        target_vector = labeling_df[[cell_type]].values
        res = scipy.stats.spearmanrho(expression_levels_df, target_vector, axis=0)
        results.append(res.statistic)

    result_matrix = np.column_stack(results)

    results = pd.DataFrame(data=result_matrix,
                      columns=labeling_df.columns,
                      index=expression_levels_df.columns)

    return results
