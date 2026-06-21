import numpy as np
import scanpy as sc
from scipy.stats import rankdata

import config

LAYER_NAME_RAW_COUNTS = "raw_counts"
LAYER_NAME_NORMALIZED_COUNTS = "normalized_counts"
LAYER_NAME_LOGARITHMIZED = "logarithmized"
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
    sc.pp.scale(dataset, max_value=10)


def apply_basic_filtering_in_place(rna,
                                   min_genes=200,
                                   max_pct_mito=20.0):
    """
    Data is already filtered to begin with.
    The filtering here is for extra caution.

    Keep only genes with:
    1. min_genes count
    2.

    """
    keep = (rna.obs["n_genes_by_counts"] >= min_genes) & (
        rna.obs["pct_counts_mito"] <= max_pct_mito)
    return rna[keep].copy()


def annotate_highly_variable_genes(dataset, n_top=config.N_HVG):
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


def get_highly_variable_genes(dataset, n_top=config.N_HVG):
    return dataset.var_names[dataset.var["highly_variable"]].tolist()


def rank_zscore(X: np.ndarray) -> np.ndarray:
    """Per-column average-rank transform, then z-score. Returns float32."""
    R = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        R[:, j] = rankdata(X[:, j], method="average")
    R -= R.mean(axis=0, keepdims=True)
    sd = R.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (R / sd).astype(np.float32)


def build_matrix_of_interest(dataset):
    """Shared (cells x HVG) rank-transformed/z-scored matrix"""

    X = dataset.layers[LAYER_NAME_LOGARITHMIZED]

    # Layers on the main dataset should always be in a sparse representation
    # Convert to dense representation
    X = np.asarray(X.todense())
    return rank_zscore(X)


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
