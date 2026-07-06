from typing import cast

import numpy as np
import pandas as pd
import scanpy as sc
import mudata as mu
from anndata import AnnData
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

from src import config
import logging

log = logging.getLogger(__file__)

LAYER_NAME_RAW_COUNTS = "raw_counts"
LAYER_NAME_NORMALIZED_COUNTS = "normalized_counts"
LAYER_NAME_LOGARITHMIZED = "logarithmized"
LAYER_NAME_RANK_TRANSFORMED = "rank_transformed"
LAYER_NAME_SCALED = "scaled"
OBSM_NAME_PCA = "X_pca"
OBSM_NAME_PCA_HARMONY = "X_pca_harmony"
OBSM_NAME_UMAP = "X_umap"
OBSM_NAME_UMAP_HARMONY = "X_umap_harmony"

LABLES_TO_DROP = ["Doublet"]


def calculate_qc_metrics_in_place(dataset):
    # Mitochondrial genes
    dataset.var["mt"] = dataset.var["gene_name"].str.startswith("MT-")
    sc.pp.calculate_qc_metrics(dataset,
                               qc_vars=["mt"],
                               inplace=True,
                               percent_top=None,
                               log1p=False)


def apply_basic_filtering(rna_dataset: AnnData,
                          level: str,
                          min_gene_per_cell=200,
                          min_cells_per_gene=3,
                          max_pct_mito=20.0) -> AnnData:
    """Data is already filtered to begin with.
    The filtering here is for extra caution.

    Keep only cells (rows) that:
    1. have more than min_gene_per_cell unique expressed genes (default: 200)
    2. don't have too much mitochondrial genes (pct_counts_mito < max_pct_mito)
    3. Don't have specific labels (e.g: Doublets)

    Keep only genes that:
    1. are expressed in more than min_cells_per_gene (default: 3)

    """

    sc.pp.filter_genes(rna_dataset, min_cells=min_cells_per_gene)
    sc.pp.filter_cells(rna_dataset, min_genes=min_gene_per_cell)

    # Filter out cells with high mitochondrial expression levels
    rna_dataset = rna_dataset[
        rna_dataset.obs['pct_counts_mt'] < max_pct_mito, :]

    # Filter out cells with unwanted labels
    rna_dataset = rna_dataset[~rna_dataset.obs[level].isin(LABLES_TO_DROP), :]

    return rna_dataset.copy()


def rank_zscore(matrix: np.ndarray) -> np.ndarray:
    """Per-gene average-rank transform, then z-score. Returns float32.

    This is the **shared feature transform**: every marker-scoring method (Spearman,
    partial correlation, MI, and the MLP/IG) must see exactly this matrix so that
    differences in their gene rankings reflect the statistical measure and not
    different inputs. The average-rank step also collapses the many dropout zeros of
    a gene to a single shared rank, mitigating the zero-inflation confound (important
    for MI, whose Ross-2014 kNN estimator is sensitive to it).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    ranked = np.empty_like(matrix)
    for j in range(matrix.shape[1]):
        ranked[:, j] = rankdata(matrix[:, j], method="average")
    ranked -= ranked.mean(axis=0, keepdims=True)
    sd = ranked.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (ranked / sd).astype(np.float32)


def apply_rank_transform_to_split_data(training_rna_dataset: AnnData,
                                       test_rna_dataset: AnnData):
    """Store the shared rank-transformed/z-scored matrix in ``LAYER_NAME_RANK_TRANSFORMED``.

    Ranks are computed *within* each split independently (rank is a relative,
    within-sample quantity), mirroring how each dataset is scored on its own cells.
    This layer is the correct input for **all four** methods; use it instead of
    ``LAYER_NAME_SCALED`` in training/tuning/scoring.
    """
    training_rna_dataset.layers[LAYER_NAME_RANK_TRANSFORMED] = rank_zscore(
        training_rna_dataset.to_df().values)
    test_rna_dataset.layers[LAYER_NAME_RANK_TRANSFORMED] = rank_zscore(
        test_rna_dataset.to_df().values)


def apply_scaling_to_split_data(training_rna_dataset: AnnData,
                                test_rna_dataset: AnnData):
    """Scale the data and save in a layer. This is needed later on.

        The scaling goes as follows:
            1. Find the parameters (mean, sd) of the training data (fit)
            2. Transform the training data and save in a layer (do not change the main matrix)
            3. Transform the test data *based on the parameters from the training data
            and save in a layer (do not change the main matrix)
    """

    scaler = StandardScaler()

    # Fit, scale and save the training data
    training_data_scaled = scaler.fit_transform(training_rna_dataset.to_df())
    training_rna_dataset.layers[LAYER_NAME_SCALED] = training_data_scaled

    # Scale and save the test data based on the parameters from the training
    # dataset.
    test_data_scaled = scaler.transform(test_rna_dataset.to_df())
    test_rna_dataset.layers[LAYER_NAME_SCALED] = test_data_scaled


def apply_scaling(rna_dataset: AnnData):
    """Single-dataset variant of :func:`apply_scaling_to_split_data`: fit and
    transform the SAME dataset (no separate held-out set to fit against). Use this
    when only a training split is loaded (e.g. MLP training, which does not load
    the test split), so ``LAYER_NAME_SCALED`` exists before ``to_df(layer=...)``
    is called on it.
    """
    scaler = StandardScaler()
    rna_dataset.layers[LAYER_NAME_SCALED] = scaler.fit_transform(
        rna_dataset.to_df())


def apply_basic_filtering_to_split_data(training_rna_dataset_filtered: AnnData,
                                        test_rna_dataset_filtered: AnnData,
                                        level: str = config.DEFAULT_LEVEL,
                                        min_gene_per_cell=200,
                                        min_cells_per_gene=3,
                                        max_pct_mito=20.0) -> tuple[
    AnnData, AnnData]:
    """Filtering split data

    After spliting, we deal with subsets of the observations (the cells).
    It might be that in the training dataset, some genes that were expressed in
    the full dataset, are now only expressed in fewer cells than allowed.

    Such genes don't contriute any valuable information to the analysis of that
    specific split and may/should be filtered out to reduce computational load.

    The same applies for cells per genes: some genes might not be expressed in
    enough cells to pass the threshold.

    In such filtering, we establish the list of genes to filter solely
    based on the training set. We then subset the test dataset accordingly.

    Keep only cells (rows) that:
    1. have more than min_gene_per_cell unique expressed genes (default: 200)
    2. don't have too much mitochondrial genes (pct_counts_mito < max_pct_mito)
    3. Don't have specific labels (e.g: Doublets)

    Keep only genes that:
    1. are expressed in more than min_cells_per_gene (default: 3)

    """

    sc.pp.filter_genes(training_rna_dataset_filtered,
                       min_cells=min_cells_per_gene)
    sc.pp.filter_cells(training_rna_dataset_filtered,
                       min_genes=min_gene_per_cell)

    training_rna_dataset_filtered = training_rna_dataset_filtered[
        training_rna_dataset_filtered.obs['pct_counts_mt'] < max_pct_mito, :]
    training_rna_dataset_filtered = training_rna_dataset_filtered[
        ~training_rna_dataset_filtered.obs[level].isin(LABLES_TO_DROP), :]

    test_genes_to_keep = test_rna_dataset_filtered.var_names.isin(
        training_rna_dataset_filtered.var_names)

    test_rna_dataset_filtered = test_rna_dataset_filtered[:, test_genes_to_keep]

    # We know that the datasets are AnnData objects
    return training_rna_dataset_filtered.copy(), test_rna_dataset_filtered.copy()


def annotate_highly_variable_genes(dataset: AnnData,
                                   n_top: int = config.N_TOP_HVGs):
    """This method extends the gene (var) annotations in place.

    See the documentation for details about the added annotations:
        https://scanpy.scverse.org/en/stable/generated/scanpy.pp.highly_variable_genes.html

    """
    sc.pp.highly_variable_genes(dataset,
                                n_top_genes=n_top,
                                flavor="seurat_v3",
                                layer=LAYER_NAME_RAW_COUNTS,
                                subset=False)


def get_highly_variable_genes(dataset: AnnData):
    return dataset.var["gene_name"][dataset.var["highly_variable"]]


def build_target_df(dataset,
                    level) -> pd.DataFrame:
    """Creates a binary one-hot encoded matrix mapping cells to their specific cell types.

        Loops through all categories at the specified annotation level and creates
        a 1-or-0 mask array for each cell type.

        Args:
            dataset: The AnnData or MuData object containing the cell annotations.
            level: The column name in dataset.obs that holds the cell type labels.

        Returns:
            A DataFrame where rows are cell barcodes, columns are cell types,
            and values are 1 if the cell belongs to that type (0 otherwise).
        """
    cell_types = np.unique(dataset.obs[level].values)

    target_vectors = []

    for cell_type in cell_types:
        mask = (dataset.obs[level] == cell_type).astype(np.uint8)
        target_vectors.append(mask)

    target_matrix = np.column_stack(target_vectors)

    return pd.DataFrame(
        target_matrix,
        index=dataset.obs_names,
        columns=cell_types
    )
