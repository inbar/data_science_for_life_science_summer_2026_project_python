"""Protein-derived ground truth D_c.

For each cell type we take the surface proteins that are *elevated* in that type
versus the rest (one-vs-rest Wilcoxon on CLR-normalised ADT), then map those
proteins to their encoding gene(s) via the molecule-level curation in
:mod:`protein_gene_map`. The four RNA methods never see the ADT modality, so this
keeps the evaluation a genuine cross-modal test.

Only molecular fact enters the protein->gene step; no differential-expression
prior is used, so D_c is independent of all four methods. Cell types with fewer
than ``min_drivers`` mapped genes (after intersecting the scored gene universe)
are excluded downstream.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from notebooks.notebook import cell_types
from src import config

from src.adt.preprocessing import LAYER_NAME_LOGARITHMIZED
from src.mappings import map_protein_to_genes


def compute_marker_proteins(dataset,
                            group_by,
                            pval_cutoff=0.05,
                            log2fc_min=0.0,
                            top_k=10):
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

    df.sort_values(by=["group", "scores"],
                   ascending=False,
                   inplace=True)

    return df


def build_ground_truth(adt_dataset,
                       group_by,
                       genes_of_interest,
                       pval_cutoff=0.05,
                       log2fc_min=0.0,
                       top_k=10):
    """
    Build the ground truth mapping: Cell -> set of drive genes

    The construction goes as follows:
    1. Compute the set of top marker (one vs. rest, differentially significnt)
     proteins for each cell
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

    # Filter the gene list for each cell by genes of interest
    marker_proteins["genes_of_interest"] = (
        marker_proteins["genes"]
        .explode()
        .loc[lambda gene: gene.isin(genes_of_interest)]
        .groupby(level=0)
        .agg(lambda gene_series: gene_series.to_list())
    )

    return cell_type_to_driver_gene_mapping, marker_proteins


# TODO
def summarise_driver_genes(cell_type_to_driver_gene_mapping):
    df = pd.DataFrame(
        {
            "cell_type": list(cell_type_to_driver_gene_mapping),
            "n_drivers": [len(v) for v in
                          cell_type_to_driver_gene_mapping.values()],
            "drivers": cell_type_to_driver_gene_mapping.values()
        }
    ).sort_values("cell_type", ascending=False).reset_index(drop=True)

    return df
