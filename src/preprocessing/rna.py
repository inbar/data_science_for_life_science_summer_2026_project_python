import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.preprocessing import StandardScaler

from src import config
from src import logs

log = logs.get_logger()

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


def scale_to_layer(training_data: AnnData,
                   test_data: AnnData):
    """ Scale the data and save in a layer. This is needed later on.

        The scaling goes as follows:
            1. Find the parameters (mean, sd) of the training data (fit)
            2. Transform the training data and save in a layer (do not change the main matrix)
            3. Transform the test data *based on the parameters from the training data
            and save in a layer (do not change the main matrix)
    """
    training_data_rna = training_data["rna"]
    test_data_rna = test_data["rna"]

    scaler = StandardScaler()

    # Fit, scale and save the training data
    training_data_scaled = scaler.fit_transform(training_data_rna.to_df())
    training_data_rna.layers[LAYER_NAME_SCALED] = training_data_scaled

    # Scale and save the test data based on the parameters from the training
    # dataset.
    test_data_scaled = scaler.transform(test_data_rna.to_df())
    test_data_rna.layers[LAYER_NAME_SCALED] = test_data_scaled


def apply_basic_filtering(dataset: AnnData,
                          level: str,
                          min_gene_count=200,
                          max_pct_mito=20.0):
    """Data is already filtered to begin with.
    The filtering here is for extra caution.

    Keep only genes with:
    1. n_genes_by_counts > min_genes
    2. pct_counts_mito < max_pct_mito
    3. Not in [labels to drop]

    """

    sc.pp.filter_cells(dataset, min_counts=min_gene_count)

    dataset = dataset[dataset.obs['pct_counts_mt'] < max_pct_mito, :]
    dataset = dataset[~dataset.obs[level].isin(LABLES_TO_DROP), :]

    return dataset.copy()


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


