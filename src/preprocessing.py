"""Preprocessing: QC, normalization, HVGs, the shared feature matrix, and
embeddings (PCA / UMAP / Harmony).

The single most important object produced here is the **shared rank-transformed
matrix** ``X_ranked`` (cells x HVGs): per-gene average-rank transform followed by
z-scoring. Every one of the four methods sees exactly this matrix, so differences
in their rankings reflect the statistical measure, not different inputs. The
average-rank transform also collapses the many dropout zeros of a gene to a single
shared rank, mitigating the zero-inflation confound that the pitch flags for MI.
"""
from __future__ import annotations

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import rankdata

import config


# TODO: delete

# --------------------------------------------------------------------------- #
# QC + normalization
# --------------------------------------------------------------------------- #
def compute_qc(rna):
    rna.var["mito"] = rna.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mito"], inplace=True,
                               percent_top=None, log1p=False)
    return rna


def basic_filter(rna, min_genes=200, max_pct_mito=20.0):
    """Light QC filter (data is already author-QC'd; this is a guard)."""
    keep = (rna.obs["n_genes_by_counts"] >= min_genes) & \
           (rna.obs["pct_counts_mito"] <= max_pct_mito)
    return rna[keep].copy()


def normalize_rna(rna):
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    rna.layers["lognorm"] = rna.X.copy()
    return rna


def normalize_adt(adt):
    """Centered-log-ratio across proteins within each cell (ADT standard)."""
    X = adt.X
    X = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
    adt.layers["counts"] = sp.csr_matrix(X)
    logx = np.log1p(X)
    gm = logx.mean(axis=1, keepdims=True)
    adt.layers["clr"] = (logx - gm).astype(np.float32)
    adt.X = adt.layers["clr"]
    return adt



def select_hvg(rna, n_top=config.TOP_K_HVG):
    """HVGs -> shared gene universe for all methods.

    Prefer the count-based ``seurat_v3`` flavor; fall back to the log-normalised
    ``seurat`` flavor if the optional ``skmisc`` dependency is unavailable.
    """
    try:
        sc.pp.highly_variable_genes(rna, n_top_genes=n_top, flavor="seurat_v3",
                                    layer="counts", subset=False)
    except ImportError:
        sc.pp.highly_variable_genes(rna, n_top_genes=n_top, flavor="seurat",
                                    subset=False)  # runs on current (lognorm) X
    return rna.var_names[rna.var["highly_variable"]].tolist()


# --------------------------------------------------------------------------- #
# Shared rank-transformed matrix
# --------------------------------------------------------------------------- #
def rank_zscore(X: np.ndarray) -> np.ndarray:
    """Per-column average-rank transform, then z-score. Returns float32."""
    R = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        R[:, j] = rankdata(X[:, j], method="average")
    R -= R.mean(axis=0, keepdims=True)
    sd = R.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (R / sd).astype(np.float32)


def build_shared_matrix(rna, genes_of_interest):
    """Shared (cells x HVG) rank-transformed/z-scored matrix + gene names."""

    gene_mask = rna.var["gene_name"].isin(sorted(genes_of_interest))

    subset = rna[:, gene_mask]
    X = subset.layers["lognorm"]
    # Layers on the main dataset should always be in a sparse representation
    # Convert to dense

    X = np.asarray(X.todense())
    return rank_zscore(X), list(genes_of_interest)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def embed(rna, hvg_genes, donor_key=config.DONOR_KEY, n_pcs=config.N_PCS,
          seed=config.SEED, run_harmony=True):
    """PCA + UMAP on HVGs, and (optionally) Harmony batch correction over donors.

    Stores: X_pca, X_umap, and (if harmony) X_pca_harmony, X_umap_harmony.
    """
    work = rna[:, hvg_genes].copy()
    work.X = work.layers["lognorm"].copy()
    sc.pp.scale(work, max_value=10)
    # 'randomized' avoids the ARPACK/LAPACK delay-load path that can clash with
    # torch's bundled MKL/OpenMP runtime on Windows.
    sc.tl.pca(work, n_comps=n_pcs, random_state=seed, svd_solver="randomized")
    rna.obsm["X_pca"] = work.obsm["X_pca"]

    sc.pp.neighbors(work, n_pcs=n_pcs, random_state=seed)
    sc.tl.umap(work, random_state=seed)
    rna.obsm["X_umap"] = work.obsm["X_umap"]

    if run_harmony:
        sc.external.pp.harmony_integrate(work, donor_key, basis="X_pca",
                                         adjusted_basis="X_pca_harmony")
        rna.obsm["X_pca_harmony"] = work.obsm["X_pca_harmony"]
        sc.pp.neighbors(work, use_rep="X_pca_harmony", random_state=seed)
        sc.tl.umap(work, random_state=seed)
        rna.obsm["X_umap_harmony"] = work.obsm["X_umap"]
    return rna
