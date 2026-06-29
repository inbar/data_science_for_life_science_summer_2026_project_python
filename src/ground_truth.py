from __future__ import annotations

import pandas as pd
import scanpy as sc

from preprocessing.adt import LAYER_NAME_LOGARITHMIZED
from src.mappings import map_protein_to_genes

def compute_marker_proteins(dataset,
                            group_by,
                            pval_cutoff=0.05,
                            log2fc_min=0.0,
                            top_k=10)-> pd.DataFrame:
    """Find top_k elevated proteins per cell type.

    To find elevated proteins we apply a one-vs-rest differential analysis
    proteins per cell type using Wilcoxon method on the logarithmized data
    (not CLR, which typically contains negative values).

    The results are filtered to have:
    1. `score` < 0
    2. `pval` < pval_cutoff
    3. `log2fc` < log2fc_min

    These proteins will be then considered as markers for each cell type.
    """

    # Calculate wilcoxon one vs. rest differential expression for the *proteins*
    # using logarithmized data (not CLR, which typically contains negative values).
    #
    # Pick the top_k proteins.
    sc.tl.rank_genes_groups(dataset,
                            use_raw=False,
                            layer=LAYER_NAME_LOGARITHMIZED,
                            groupby=group_by,
                            n_genes=top_k,
                            method="wilcoxon")

    # Extract the results while applying the cutoffs.
    df = sc.get.rank_genes_groups_df(dataset,
                                     group=None,
                                     gene_symbols="protein_name",
                                     log2fc_min=log2fc_min,
                                     pval_cutoff=pval_cutoff)

    # Throw away any entries with scores < 0
    df = df[df["scores"] > 0]

    return df


def build_ground_truth(adt_dataset,
                       group_by,
                       genes_of_interest,
                       pval_cutoff=0.05,
                       log2fc_min=0.0,
                       top_k=10) -> dict[str, set]:
    """
    Build the ground truth mapping: Cell -> set of drive genes

    The construction goes as follows:
    1. Compute the set of top marker proteins (one vs. rest, differentially significnt)
       for each cell
    2. For each marker protein, fetch a list of its encoding genes.
    3. The set of all encoding genes for each cell define the cells "driver gene"
      set.

    return:
        cell_type_to_driver_gene_mapping: dict[cell_type -> set(driver genes)]
            a mapping of each cell to the list of its driver genes

        marker_proteins: Dataframe
            long DataFrame of elevated proteins with their mapped genes.
    """

    marker_proteins = compute_marker_proteins(dataset=adt_dataset,
                                              group_by=group_by,
                                              pval_cutoff=pval_cutoff,
                                              log2fc_min=log2fc_min,
                                              top_k=top_k)

    marker_proteins["genes"] = marker_proteins["protein_name"].apply(
        map_protein_to_genes)

    cell_type_to_driver_gene_mapping = {}
    for cell_type, df in marker_proteins.groupby("group", observed=True):
        drivers = set(df["genes"].explode())
        drivers = drivers.intersection(genes_of_interest)
        cell_type_to_driver_gene_mapping[cell_type] = drivers

    return cell_type_to_driver_gene_mapping


# TODO
def pretty_print_ground_truth(ground_truth_object):
    """Pretty print a ground truth object created with build_ground_truth()
    """
    df = pd.DataFrame(
        {
            "cell_type": list(ground_truth_object),
            "n_drivers": [len(v) for v in
                          ground_truth_object.values()],
            "drivers": ground_truth_object.values()
        }
    ).sort_values("cell_type", ascending=False).reset_index(drop=True)

    return df
