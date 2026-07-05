from collections import defaultdict
from io import StringIO

import pandas as pd
import requests
# Test one vs Rest an use The wilcoxon ranksum test in scipy
from scipy.stats import ranksums


def ranksumtest_for_groundtruth(adata_adt,
                                top_proteins=False):  # top_proteins = True then we will jsut take the top 20 ADTs
    rows = []
    adt_df = pd.DataFrame(adata_adt.X.toarray(), index=adata_adt.obs_names,
                          columns=adata_adt.var_names)
    # Lets loop through every cell type
    for celltype in adata_adt.obs[
        "celltype.l1"].unique():  # Here we can change the level of granularity.

        # Filter for the celltype we are looking at
        filtered_for_celltype = adata_adt.obs["celltype.l1"] == celltype

        # For every celltype we will loop through every surface_protein
        for surface_protein in adata_adt.var_names:
            adt_in_celltype = adt_df.loc[
                filtered_for_celltype.values, surface_protein]
            adt_in_rest = adt_df.loc[
                ~filtered_for_celltype.values, surface_protein]

            # Now we will do the ranksums test
            score_wilc_ranksum_test, p_val = ranksums(adt_in_celltype,
                                                      adt_in_rest)

            # To filter later, we also need the mean difference
            mean_difference = adt_in_celltype.mean() - adt_in_rest.mean()

            # Now we append the row to rows
            rows.append(
                {"Celltype": celltype, "Surface_Protein": surface_protein,
                 "Score_Wilcox_Ranksum": score_wilc_ranksum_test,
                 "P_Value": p_val, "Mean_Difference": mean_difference})

    surface_protein_marker_df = pd.DataFrame(rows)

    # Now we filter the Proteins which are are significantly more present on a specific celltype -> A WicoxRankSum Score > 0 and a mean diff > 0
    surface_protein_marker_df = surface_protein_marker_df[
        (surface_protein_marker_df["Score_Wilcox_Ranksum"] > 0) & (
                surface_protein_marker_df["Mean_Difference"] > 0)].copy()

    if top_proteins:
        # Take the top 20 surface proteins
        top_surface_proteins = (
            surface_protein_marker_df
            .sort_values(["Score_Wilcox_Ranksum"], ascending=[False])
            .groupby("Celltype")
            .head(20)
            .reset_index(drop=True))

        return top_surface_proteins

    return surface_protein_marker_df


def search_uniprot_protein_to_gene(protein_names, organism_id=9606,
                                   reviewed_only=True):
    all_results = []

    for protein in protein_names:
        query = f'(protein_name:"{protein}" OR gene:{protein}) AND organism_id:{organism_id}'

        if reviewed_only:
            query += " AND reviewed:true"

        url = "https://rest.uniprot.org/uniprotkb/search"

        params = {
            "query": query,
            "fields": "accession,id,protein_name,gene_names,organism_name,reviewed",
            "format": "tsv",
            "size": 10
        }

        r = requests.get(url, params=params)
        r.raise_for_status()

        if r.text.strip():
            df = pd.read_csv(StringIO(r.text), sep="\t")
            df.insert(0, "Surface_Protein", protein)
            all_results.append(df)
        else:
            all_results.append(pd.DataFrame({
                "Surface_Protein": [protein],
                "Entry": [None],
                "Gene Names": [None]
            }))

    return pd.concat(all_results, ignore_index=True)


def load_groundtruth_as_dict(path_to_groundtruth_csv):
    groundtruth = pd.read_csv(path_to_groundtruth_csv)
    protein_to_genes = defaultdict(list)

    for _, row in groundtruth.iterrows():
        protein = row["Surface_Protein"]
        genes = row["Gene Names"]

        if pd.isna(genes):
            continue

        protein_to_genes[protein].extend(genes.split())

    # Duplikate entfernen, Reihenfolge behalten
    protein_to_genes = {
        protein: list(dict.fromkeys(genes))
        for protein, genes in protein_to_genes.items()
    }
    return protein_to_genes


def create_groundtruth(adt_dataset):
    top_surface_proteins_with_ranksum = ranksumtest_for_groundtruth(adt_dataset)
    ground_truth_df = search_uniprot_protein_to_gene(
        top_surface_proteins_with_ranksum["Surface_Protein"].unique())
    gt_merged = top_surface_proteins_with_ranksum[
        ["Celltype", "Surface_Protein"]].drop_duplicates().merge(
        ground_truth_df[["Surface_Protein", "Gene Names"]],
        on="Surface_Protein", how="left")
    gt_merged = gt_merged.dropna()
    gt_merged.to_csv("groundtruth.csv", index=False)

