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

from .protein_gene_map import map_protein_to_genes


def elevated_proteins(adt, level: str, padj: float = 0.05,
                      min_lfc: float = 0.0, top_k: int | None = 10):
    """One-vs-rest elevated proteins per cell type (returns long DataFrame).

    Decision / rationale
    --------------------
    With ~25k cells every protein is "significant", so the p-value cannot select
    markers. The directional **Wilcoxon score** is the reliable signal (scanpy's
    log-fold-change on CLR data is unstable and can be large even for proteins
    with a *negative* score). We therefore require ``score > 0`` and
    ``padj < padj``, rank by score, and keep only the ``top_k`` strongest proteins
    per cell type. This keeps D_c to specific canonical markers and excludes the
    weakly cross-reactive tail (e.g. platelet antibodies on monocyte aggregates).
    """
    a = ad.AnnData(X=np.asarray(adt.layers["clr"]), obs=adt.obs[[level]].copy(),
                   var=adt.var.copy())
    a.obs[level] = a.obs[level].astype("category")
    sc.tl.rank_genes_groups(a, groupby=level, method="wilcoxon")
    res = a.uns["rank_genes_groups"]
    groups = res["names"].dtype.names
    rows = []
    for g in groups:
        df = pd.DataFrame({
            "celltype": g,
            "protein": res["names"][g],
            "score": res["scores"][g],
            "lfc": res["logfoldchanges"][g],
            "padj": res["pvals_adj"][g],
        })
        df = df[(df["padj"] < padj) & (df["lfc"] > min_lfc) & (df["score"] > 0)]
        df = df.sort_values("score", ascending=False)
        if top_k is not None:
            df = df.head(top_k)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def build_ground_truth(adt, level: str, gene_universe=None, **kwargs):
    """Return (drivers, details).

    drivers : {cell_type -> set(driver gene symbols)} (optionally restricted to
              ``gene_universe``, i.e. the HVGs the methods actually rank).
    details : long DataFrame of elevated proteins with their mapped genes.
    """
    elev = elevated_proteins(adt, level, **kwargs)
    elev["genes"] = elev["protein"].apply(map_protein_to_genes)
    universe = set(gene_universe) if gene_universe is not None else None

    drivers: dict[str, set[str]] = {}
    for ct, sub in elev.groupby("celltype"):
        genes = set().union(*sub["genes"]) if len(sub) else set()
        if universe is not None:
            genes &= universe
        drivers[ct] = genes
    elev["genes_in_universe"] = elev["genes"].apply(
        lambda gs: [g for g in gs if universe is None or g in universe])
    return drivers, elev


def summarise_drivers(drivers: dict[str, set[str]]) -> pd.DataFrame:
    return (pd.DataFrame({"celltype": list(drivers),
                          "n_drivers": [len(v) for v in drivers.values()],
                          "drivers": [", ".join(sorted(v)) for v in drivers.values()]})
            .sort_values("n_drivers", ascending=False).reset_index(drop=True))
