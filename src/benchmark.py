"""Orchestrates the 2x2 benchmark: run every method on every cell type and score
recovery of the protein-derived driver set with AUC_rel.

The three closed-form scorers (Spearman, partial correlation, KSG-MI) are computed
here from the shared rank-transformed matrix. The fourth method's per-gene scores
(Integrated Gradients on the MLP) are passed in precomputed, because they require a
trained model (see :mod:`mlp`). Keeping all four on the identical gene universe and
the identical one-vs-rest indicator is what makes the comparison fair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import scorers
from .metric import auc_rel


def compute_mi_parallel(X_ranked, labels, celltypes, n_jobs=-1,
                        n_neighbors=3, seed=0, max_cells=10000) -> dict[str, np.ndarray]:
    """KSG mutual information per gene for each cell type, computed in parallel.

    ``mutual_info_classif`` is single-threaded and its kNN cost grows with the
    number of cells, so we (a) estimate MI on a fixed random cell subsample of
    size ``max_cells`` (MI is an estimator; ~10k cells gives a stable estimate
    and the *same* subsample is used for every cell type, keeping the comparison
    internally consistent), and (b) parallelise across the one-vs-rest cell-type
    targets (the expensive dimension). The other three methods still use all
    cells. Set ``max_cells=None`` to use every cell.
    """
    labels = np.asarray(labels)
    if max_cells is not None and X_ranked.shape[0] > max_cells:
        idx = np.random.default_rng(seed).choice(X_ranked.shape[0], max_cells,
                                                  replace=False)
        Xm, lab = X_ranked[idx], labels[idx]
    else:
        Xm, lab = X_ranked, labels

    def one(ct):
        y = (lab == ct).astype(int)
        return ct, scorers.mi_ksg_scores(Xm, y, n_neighbors, seed)
    return dict(Parallel(n_jobs=n_jobs)(delayed(one)(ct) for ct in celltypes))


def per_gene_scores(
    X_ranked: np.ndarray,
    y: np.ndarray,
    mi_vector: np.ndarray | None = None,
    ig_vector: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-gene marker scores for one binary one-vs-rest target.

    Spearman and partial correlation are cheap and computed here; MI (slow) and
    IG (needs a trained model) are passed in precomputed. Sign convention:
    correlation/partial-correlation/IG are signed (markers positively
    associated); MI is non-negative by construction.
    """
    out = {
        "spearman": scorers.spearman_scores(X_ranked, y),
        "partial_corr": scorers.partial_corr_scores(X_ranked, y),
    }
    if mi_vector is not None:
        out["mi_ksg"] = np.asarray(mi_vector, dtype=float)
    if ig_vector is not None:
        out["ig_mlp"] = np.asarray(ig_vector, dtype=float)
    return out


def run_benchmark(
    X_ranked: np.ndarray,
    genes: list[str],
    labels: np.ndarray,
    drivers: dict[str, set[str]],
    mi_attr: dict[str, np.ndarray] | None = None,
    ig_attr: dict[str, np.ndarray] | None = None,
    min_drivers: int = 2,
    mi_neighbors: int = 3,
    n_jobs: int = -1,
    seed: int = 0,
    store_scores: dict | None = None,
) -> pd.DataFrame:
    """Compute AUC_rel for every (cell type, method).

    ``mi_attr`` / ``ig_attr`` are optional precomputed {cell_type -> per-gene
    vector} maps. If ``mi_attr`` is None it is computed here (in parallel) for the
    qualifying cell types. ``store_scores`` (if given) collects the full per-gene
    score vectors for the scatter/3D plots.
    """
    gene_index = {g: i for i, g in enumerate(genes)}
    # determine qualifying cell types (|D_c| >= min_drivers and present)
    masks, valid = {}, []
    for ct, dset in drivers.items():
        present = [g for g in dset if g in gene_index]
        if len(present) < min_drivers or (labels == ct).sum() == 0:
            continue
        m = np.zeros(len(genes), dtype=bool)
        m[[gene_index[g] for g in present]] = True
        masks[ct] = m
        valid.append(ct)

    if mi_attr is None:
        mi_attr = compute_mi_parallel(X_ranked, labels, valid, n_jobs,
                                      mi_neighbors, seed)

    rows = []
    for ct in valid:
        mask = masks[ct]
        y = (labels == ct).astype(int)
        sc = per_gene_scores(X_ranked, y,
                             mi_vector=mi_attr.get(ct) if mi_attr else None,
                             ig_vector=ig_attr.get(ct) if ig_attr else None)
        if store_scores is not None:
            store_scores.setdefault(ct, {}).update(sc)
            store_scores[ct]["_driver_mask"] = mask
        for method, vec in sc.items():
            rows.append({
                "celltype": ct, "method": method,
                "auc_rel": auc_rel(vec, mask),
                "n_drivers": int(mask.sum()), "n_cells_pos": int(y.sum()),
            })
    return pd.DataFrame(rows)


def bootstrap_over_cells(X_ranked, genes, labels, drivers, n_boot=5,
                         boot_cells=3000, mi_neighbors=3, n_jobs=-1, seed=0):
    """Cell-resampling stability of the closed-form methods (Spearman, partial
    correlation, MI). Returns a per-method AUC_rel summary with percentile CIs.
    Each iteration resamples ``boot_cells`` cells with replacement."""
    from .stats import bootstrap_auc
    labels = np.asarray(labels)
    n = len(labels)

    def score_fn(rng):
        idx = rng.integers(0, n, boot_cells)
        return run_benchmark(X_ranked[idx], genes, labels[idx], drivers,
                             ig_attr=None, mi_neighbors=mi_neighbors,
                             n_jobs=n_jobs, seed=seed)

    return bootstrap_auc(score_fn, n_boot=n_boot, seed=seed)


def seed_stability_ig(X_ranked, genes, labels, drivers, mi_attr, seeds=(0, 1, 2),
                      seed=0):
    """MLP-seed stability of Integrated Gradients: retrain the classifier under
    several seeds and report ig_mlp mean AUC_rel (and test accuracy) per seed."""
    import pandas as pd
    from . import mlp as mlpmod
    labels = np.asarray(labels)
    rows = []
    for s in seeds:
        t = mlpmod.train_mlp(X_ranked, labels, seed=s)
        attr = mlpmod.integrated_gradients(t, X_ranked, labels, seed=s)
        classes = list(t.classes)
        ig = {ct: attr[:, classes.index(ct)] for ct in classes}
        r = run_benchmark(X_ranked, genes, labels, drivers, mi_attr=mi_attr,
                          ig_attr=ig, seed=s)
        r = r[r.method == "ig_mlp"]
        rows.append({"seed": s, "mean_auc_rel": float(r["auc_rel"].mean()),
                     "test_acc": float(t.test_acc)})
    return pd.DataFrame(rows)


def recovery_at_k_table(store_scores: dict,
                        methods=("spearman", "partial_corr", "mi_ksg", "ig_mlp"),
                        ks=(1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100)) -> pd.DataFrame:
    """Tidy [celltype, method, k, recovery] table of recall@k from stored scores."""
    from .metric import recovery_at_k
    rows = []
    for ct, d in store_scores.items():
        mask = d["_driver_mask"]
        for m in methods:
            if m not in d:
                continue
            for k in ks:
                rows.append({"celltype": ct, "method": m, "k": int(k),
                             "recovery": recovery_at_k(d[m], mask, k)})
    return pd.DataFrame(rows)


def scores_to_long(store_scores: dict, genes: list[str],
                   methods=("spearman", "partial_corr", "mi_ksg", "ig_mlp"),
                   standardize: bool = True) -> pd.DataFrame:
    """Tidy per-(gene, cell type) score table for the scatter-matrix / 3D plots.

    With ``standardize`` each method's scores are z-scored within a cell type so
    that genes/cell types are comparable when pooled. A boolean ``is_driver``
    column marks the protein-derived drivers.
    """
    frames = []
    for ct, d in store_scores.items():
        mask = d["_driver_mask"]
        cols = {"celltype": ct, "gene": genes, "is_driver": mask}
        for m in methods:
            if m not in d:
                continue
            v = np.asarray(d[m], dtype=float)
            if standardize:
                sd = np.nanstd(v)
                v = (v - np.nanmean(v)) / (sd if sd else 1.0)
            cols[m] = v
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)
