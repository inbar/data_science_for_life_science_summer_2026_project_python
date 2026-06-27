import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import pyplot as plt

from logs import get_logger

log = get_logger()

def calculate_validation_data(results_df: pd.DataFrame,
                              top_k_genes=5) -> pd.DataFrame:
    spearman_results_long_df = results_df.reset_index().melt(
        id_vars='gene_name',
        var_name='cell_type',
        value_name='score'
    )

    top_gene_df = (spearman_results_long_df
                   .sort_values(by="score", ascending=False)
                   .groupby("cell_type")
                   .head(top_k_genes))

    return top_gene_df


def plot(top_gene_df: pd.DataFrame,
         rna_dataset: AnnData,
         level: str):
    # Prepare the data
    # List of gene names
    # Make sure the data is sorted
    top_gene_df.sort_values(by=["cell_type", "score"])
    top_gene_list = top_gene_df["gene_name"].unique()

    # Subset the rna dataset
    top_gene_mask = rna_dataset.var["gene_name"].isin(top_gene_list)
    rna_dataset_subset = rna_dataset[:, top_gene_mask]
    rna_dataset_subset_df = rna_dataset_subset.to_df()

    # Update column names with origial gene name
    # (column names are prefixed to be unique)
    rna_dataset_subset_df.columns = rna_dataset_subset.var['gene_name']

    # Scale column data (z-score normalization)
    col_mean = rna_dataset_subset_df.mean(axis=0)
    col_std = rna_dataset_subset_df.std(axis=0)
    normalized_df = (rna_dataset_subset_df
                     .sub(col_mean, axis=1)  # Substract mean
                     .div(col_std, axis=1))  # Divide by std

    # Eliminate NA values
    normalized_df = normalized_df.fillna(0)

    # Add cell_type column
    normalized_df["cell_type"] = rna_dataset.obs[level].values

    # Group all cell types and save the mean value
    normalized_df = normalized_df.groupby("cell_type",
                                                  observed=True).mean()

    # Transpose: cell type in the columns, genes in the rows,
    normalized_df = normalized_df.T

    # Reorder rows to match the order in the original top_gene_df
    df = normalized_df.reindex(top_gene_list)

    # Plot
    fig, ax = plt.subplots(figsize=(4, 10))

    im = ax.imshow(df,
                   aspect="auto",
                   cmap="viridis",
                   vmin=-2.5,
                   vmax=2.5)

    # X-axis: Aggregated Cell Types
    x_vals = df.columns
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels(x_vals, rotation=45, ha='right', fontsize=6)


    # Y-axis: Top Driver Genes
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=5)

    # Colorbar matching your settings
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Mean Expression Level for Cell Type")

    ax.set_title("Spearman Method Validation Heatmap (Mean)", pad=15,
                 fontsize=12, weight='bold')
