"""Driver-recovery metric (AUC_rel).

Each marker-scoring method emits one scalar score per gene for a given cell type.
The protein-derived ground truth gives a driver set ``D`` (a subset of the scored
genes). We score how well the ranking recovers ``D``.

Decision / rationale
--------------------
The pitch refers to the "rank-based driver-recovery metric of CellRank 2
(AUC_rel), parameter-free, between 0.5 and 1.0". CellRank ranks genes by score,
draws the cumulative-recovery curve (fraction of true drivers found within the
top-k genes) and reports its area normalised so that a random ranking scores 0.5
and a perfect ranking ~1.0.

For |D| << N that normalised recovery-curve area is mathematically the
**Mann-Whitney / ROC-AUC** of the score discriminating driver vs non-driver
genes: it is parameter-free, bounded in [0, 1] with 0.5 = random and 1.0 =
perfect, and defined on exactly the per-gene score every method produces. We
therefore implement AUC_rel as that rank statistic (ties handled by midranks),
which is the robust, dependency-free realisation of the CellRank metric.

We also expose the explicit cumulative-recovery curve for plotting.
"""
from __future__ import annotations

import numpy as np


def _as_rank_auc(scores: np.ndarray, is_driver: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic with midrank tie handling.

    Equivalent to the normalised area under the cumulative driver-recovery
    curve. Returns NaN if the set is degenerate (all or no drivers).
    """
    scores = np.asarray(scores, dtype=float)
    is_driver = np.asarray(is_driver, dtype=bool)
    n_pos = int(is_driver.sum())
    n_neg = int((~is_driver).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(scores, kind="mergesort")
    ranked = scores[order]
    # midranks (average ranks for ties), 1-based
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1] == ranked[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_of = np.empty(len(scores), dtype=float)
    rank_of[order] = ranks

    sum_pos = rank_of[is_driver].sum()
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def auc_rel(scores: np.ndarray, driver_mask: np.ndarray) -> float:
    """Relative driver-recovery AUC for one cell type / method.

    Parameters
    ----------
    scores
        Per-gene marker score (higher = more marker-like). Sign convention is
        the responsibility of the caller (e.g. signed for correlation/IG,
        magnitude for MI).
    driver_mask
        Boolean array, True where the gene is a ground-truth driver.
    """
    return _as_rank_auc(scores, driver_mask)


def recovery_at_k(scores: np.ndarray, driver_mask: np.ndarray, k: int) -> float:
    """Fraction of drivers found in the top-``k`` ranked genes (recall@k).

    Unlike the whole-curve AUC_rel, this targets *early* recovery — the regime a
    practitioner actually inspects (the top handful of genes). Returns NaN if the
    cell type has no drivers.
    """
    scores = np.asarray(scores, dtype=float)
    driver_mask = np.asarray(driver_mask, dtype=bool)
    n_drivers = int(driver_mask.sum())
    if n_drivers == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    return float(driver_mask[order][:k].sum()) / n_drivers


def recovery_curve(scores: np.ndarray, driver_mask: np.ndarray):
    """Cumulative driver-recovery curve for plotting.

    Returns
    -------
    frac_genes : np.ndarray
        Fraction of the gene list scanned (x-axis, 0..1).
    frac_recovered : np.ndarray
        Fraction of drivers recovered within that prefix (y-axis, 0..1).
    """
    scores = np.asarray(scores, dtype=float)
    driver_mask = np.asarray(driver_mask, dtype=bool)
    order = np.argsort(-scores, kind="mergesort")  # descending
    hits = driver_mask[order].astype(float)
    n = len(scores)
    n_drivers = max(int(driver_mask.sum()), 1)
    frac_genes = np.arange(1, n + 1) / n
    frac_recovered = np.cumsum(hits) / n_drivers
    return frac_genes, frac_recovered
