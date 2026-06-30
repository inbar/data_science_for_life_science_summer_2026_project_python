import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif

from src import config

from src.logs import get_logger

log = get_logger()

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
                     seed=config.DEFAULT_SEED):
    """Computes the marginal KSG Mutual Information for every gene against
    every cell type.

    KSG: A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.

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

    log.info("Computing Non-Linear/marginal scoring: Mutual Information (KSG)")

    # n_jobs=-1 will utilize all available CPU cores automatically
    n_jobs = -1

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_mi_scores)(
            cell_type,
            expression_levels_df,
            labeling_df[cell_type],
            k_neighbors,
            seed
        )
        for cell_type in labeling_df.columns
    )

    score_dict = {cell_type: scores for cell_type, scores in results}
    genes = expression_levels_df.columns
    results = pd.DataFrame(score_dict,
                           index=genes)

    return results
