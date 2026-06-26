import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
import config

from src import logs

log = logs.get_logger()

LAYER_NAME_RAW_COUNTS = "raw_counts"
LAYER_NAME_NORMALIZED_COUNTS = "normalized_counts"
LAYER_NAME_LOGARITHMIZED = "logarithmized"
LAYER_NAME_RANK_TRANSFORMED = "rank_transformed"
OBSM_NAME_PCA = "X_pca"
OBSM_NAME_PCA_HARMONY = "X_pca_harmony"
OBSM_NAME_UMAP = "X_umap"
OBSM_NAME_UMAP_HARMONY = "X_umap_harmony"


def calculate_qc_metrics_in_place(dataset):
    # Mitochondrial genes
    dataset.var["mito"] = dataset.var["gene_name"].str.startswith("MT-")
    sc.pp.calculate_qc_metrics(dataset,
                               qc_vars=["mito"],
                               inplace=True,
                               percent_top=None,
                               log1p=False)


def normalize_in_place(dataset, target_sum=1e4):
    # Store unnormalized counts in a layer
    dataset.layers[LAYER_NAME_RAW_COUNTS] = dataset.X.copy()

    # Normalize the counts for each row to `target_sum`
    sc.pp.normalize_total(dataset, target_sum=target_sum)

    # Store normalized counts in a layer
    dataset.layers[LAYER_NAME_NORMALIZED_COUNTS] = dataset.X.copy()

    # Logarithmize
    sc.pp.log1p(dataset)

    # Store normalized counts in a layer
    dataset.layers[LAYER_NAME_LOGARITHMIZED] = dataset.X.copy()


def scale_in_place(dataset):
    if sp.sparse.isspmatrix(dataset.X):
        dataset.X = dataset.X.todense()

    sc.pp.scale(dataset, max_value=10)


def apply_basic_filtering(dataset,
                          min_genes=200,
                          max_pct_mito=20.0):
    """
    Data is already filtered to begin with.
    The filtering here is for extra caution.

    Keep only genes with:
    1. n_genes_by_counts > min_genes
    2. pct_counts_mito < max_pct_mito

    """
    keep = (dataset.obs["n_genes_by_counts"] >= min_genes) & (
        dataset.obs["pct_counts_mito"] <= max_pct_mito)
    return dataset[keep].copy()


def annotate_highly_variable_genes(dataset, n_top=config.TOP_K_HVG):
    """
    This method extends the gene (var) annotations in place.

    See the documentation for details about the added annotations:
        https://scanpy.scverse.org/en/stable/generated/scanpy.pp.highly_variable_genes.html

    """
    sc.pp.highly_variable_genes(dataset,
                                n_top_genes=n_top,
                                flavor="seurat_v3",
                                layer=LAYER_NAME_RAW_COUNTS,
                                inplace=True,
                                subset=False)


def get_highly_variable_genes(dataset):
    return dataset.var["gene_name"][dataset.var["highly_variable"]].tolist()

def rank_transform_in_place(dataset):

    if sp.sparse.isspmatrix(dataset.X):
        X = dataset.X.todense()
    else:
        X = dataset.X

    X_ranked = sp.stats.rankdata(X, axis=0, method='average')
    dataset.layers[LAYER_NAME_RANK_TRANSFORMED] = X_ranked


def build_matrix_of_interest(dataset):
    """Shared (cells x HVG) rank-transformed/z-scored matrix"""

    X = dataset.layers[LAYER_NAME_LOGARITHMIZED]

    # Layers on the main dataset should always be in a sparse representation
    # Convert to dense representation
    X = np.asarray(X.todense())
    return X

def build_target_matrix(dataset,
                        level)-> pd.DataFrame:
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
    cell_types = dataset.obs[level].cat.categories.tolist()

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

def perform_pca_in_place(dataset,
                         n_pcs=config.N_PCS,
                         seed=config.SEED):
    """Perform PCA

    Stores: X_pca
    """
    sc.pp.pca(dataset, n_comps=n_pcs, random_state=seed)


def perform_pca_harmony_in_place(dataset,
                                 donor_key=config.DONOR_KEY):
    """Perform PCA with harmony batch correction

    Stores: X_pca_harmoby
    """

    sc.external.pp.harmony_integrate(dataset, donor_key)


def perform_umap_in_place(dataset,
                          n_pcs=config.N_PCS,
                          seed=config.SEED):
    """Perform UMAP

    Stores: X_umap
    """
    sc.pp.neighbors(dataset, n_pcs=n_pcs, random_state=seed)
    sc.tl.umap(dataset, random_state=seed)


def perform_umap_harmony_in_place(dataset,
                                  seed=config.SEED):
    """Perform UMAP with harmony batch correction

    Stores: X_umap_harmony
    """
    sc.pp.neighbors(dataset, use_rep=OBSM_NAME_PCA_HARMONY, random_state=seed)
    sc.tl.umap(dataset, random_state=seed, key_added=OBSM_NAME_UMAP_HARMONY)
