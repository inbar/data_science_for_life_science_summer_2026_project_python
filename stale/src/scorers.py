"""The three closed-form marker scorers of the 2x2 design.

All operate on the *same* shared feature matrix ``X`` (cells x genes), which is
rank-transformed and standardised upstream (see :mod:`preprocessing`) so that the
four methods are strictly comparable. Each returns one score per gene for a
binary one-vs-rest indicator ``y``.

    spearman_scores      linear   , marginal
    partial_corr_scores  linear   , conditional
    mi_ksg_scores        nonlinear, marginal

The fourth cell (Integrated Gradients on an MLP; nonlinear, conditional) lives in
:mod:`mlp` because it needs a trained model.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import mutual_info_classif

# TODO: refactor

# --------------------------------------------------------------------------- #
# Linear, marginal: Spearman correlation
# --------------------------------------------------------------------------- #
def spearman_scores(X_ranked: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation between each gene and the binary indicator.

    ``X_ranked`` is already rank-transformed per gene, so a Pearson correlation
    of its columns against ``y`` equals the Spearman correlation of the original
    expression against ``y``. Returns the *signed* correlation (markers are
    expected to be positively associated with their cell type).
    """
    y = np.asarray(y, dtype=float)
    yc = y - y.mean()
    Xc = X_ranked - X_ranked.mean(axis=0, keepdims=True)
    num = Xc.T @ yc
    den = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
    den[den == 0] = np.nan
    return num / den


# --------------------------------------------------------------------------- #
# Linear, conditional: partial correlation via shrinkage precision matrix
# --------------------------------------------------------------------------- #
def partial_corr_scores(X_ranked: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Partial correlation of each gene with ``y`` controlling for all genes.

    Decision / rationale
    --------------------
    The raw gene-gene covariance is ill-conditioned (dropout, collinearity,
    p ~ n), so a naive pseudoinverse gives unstable partial correlations. We use
    the **Ledoit-Wolf shrinkage** covariance of the augmented matrix [genes, y],
    which is guaranteed positive-definite and well-conditioned, invert it to a
    precision matrix ``P``, and read off the partial correlation between gene i
    and the indicator column from

        pcorr(i, y) = -P[i, y] / sqrt(P[i, i] * P[y, y]).

    The indicator is included as a variable so we obtain the gene-vs-label
    partial correlation conditioned on every other gene (a point-biserial
    partial correlation). Returns the *signed* partial correlation.
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    M = np.hstack([X_ranked, y])
    # standardise columns so shrinkage acts on a correlation-like scale
    M = M - M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    M = M / sd

    cov = LedoitWolf(assume_centered=True).fit(M).covariance_
    prec = np.linalg.inv(cov)

    yi = M.shape[1] - 1
    diag = np.diag(prec)
    pcorr = -prec[:, yi] / np.sqrt(diag * diag[yi])
    return pcorr[:-1]  # drop the indicator's self-entry


# --------------------------------------------------------------------------- #
# Nonlinear, marginal: KSG mutual information
# --------------------------------------------------------------------------- #
def mi_ksg_scores(
    X_ranked: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Kraskov-Stoegbauer-Grassberger mutual information per gene.

    Uses ``sklearn.feature_selection.mutual_info_classif`` (the KSG kNN
    estimator for a continuous feature and a discrete target) on the
    rank-transformed expression, which mitigates the zero-dropout confound.
    MI is non-negative and unsigned by construction.
    """
    return mutual_info_classif(
        X_ranked,
        np.asarray(y).astype(int),
        discrete_features=False,
        n_neighbors=n_neighbors,
        random_state=seed,
    )
