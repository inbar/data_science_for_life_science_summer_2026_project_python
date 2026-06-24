"""Statistical comparison of methods and stability analysis.

Design decision (the open question from the pitch)
--------------------------------------------------
Unit of replication = **cell type**. Each qualifying cell type contributes one
paired observation of AUC_rel per method (the four methods are evaluated on the
same cell type, so the samples are paired). We therefore use:

  * a Friedman omnibus test across the four methods (paired, non-parametric);
  * Holm-corrected Wilcoxon signed-rank tests for every method pair, with the
    matched-pairs rank-biserial effect size.

This is honest about the dependence structure (no pseudo-replication). Bootstrap
-over-cells and MLP-seed variation are reported separately as *descriptive*
stability bands, not as inferential samples.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# TODO: refactor

def pivot_auc(df: pd.DataFrame, value="auc_rel",
              index="celltype", columns="method") -> pd.DataFrame:
    """Wide cell-type x method table; rows with any NaN are dropped."""
    wide = df.pivot_table(index=index, columns=columns, values=value)
    return wide.dropna(axis=0, how="any")


def friedman_test(wide: pd.DataFrame) -> dict:
    stat, p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
    return {"statistic": float(stat), "pvalue": float(p),
            "n_celltypes": int(wide.shape[0]), "methods": list(wide.columns)}


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size for Wilcoxon signed-rank."""
    d = x - y
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    rp = r[d > 0].sum()
    rm = r[d < 0].sum()
    return float((rp - rm) / r.sum())


def pairwise_wilcoxon(wide: pd.DataFrame, method_order=None) -> pd.DataFrame:
    """Holm-corrected pairwise Wilcoxon signed-rank across method columns."""
    cols = list(method_order) if method_order else list(wide.columns)
    cols = [c for c in cols if c in wide.columns]
    rows = []
    for a, b in itertools.combinations(cols, 2):
        x, y = wide[a].values, wide[b].values
        try:
            stat, p = stats.wilcoxon(x, y, zero_method="wilcox")
        except ValueError:  # all-zero differences
            stat, p = np.nan, 1.0
        rows.append({
            "method_a": a, "method_b": b,
            "median_a": float(np.median(x)), "median_b": float(np.median(y)),
            "median_diff": float(np.median(x - y)),
            "statistic": float(stat), "pvalue": float(p),
            "rank_biserial": _rank_biserial(x, y),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["pvalue_holm"] = multipletests(out["pvalue"], method="holm")[1]
        out["significant_0.05"] = out["pvalue_holm"] < 0.05
    return out


def bootstrap_auc(score_fn, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Bootstrap AUC_rel over cells.

    ``score_fn(rng)`` must return a tidy DataFrame [celltype, method, auc_rel]
    computed on a cell-resample drawn from ``rng``. Returns per-method
    distribution summaries (mean AUC_rel across cell types, with percentile CI).
    """
    rng = np.random.default_rng(seed)
    recs = []
    for b in range(n_boot):
        d = score_fn(np.random.default_rng(rng.integers(1 << 32)))
        m = d.groupby("method")["auc_rel"].mean()
        m["_boot"] = b
        recs.append(m)
    boot = pd.DataFrame(recs).drop(columns="_boot")
    summ = pd.DataFrame({
        "mean": boot.mean(),
        "std": boot.std(),
        "ci_lo": boot.quantile(0.025),
        "ci_hi": boot.quantile(0.975),
    })
    return summ


def seed_stability(score_fn, seeds=range(5)) -> pd.DataFrame:
    """Re-run a (stochastic) scorer across seeds; return per-method mean AUC_rel
    across cell types for each seed (for e.g. MLP/IG stability)."""
    recs = []
    for s in seeds:
        d = score_fn(s)
        m = d.groupby("method")["auc_rel"].mean()
        m["seed"] = s
        recs.append(m)
    return pd.DataFrame(recs)
