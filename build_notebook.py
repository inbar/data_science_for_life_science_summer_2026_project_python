"""Generate the analysis notebook(s) from a parameterised cell list.

    python build_notebook.py            -> notebooks/marker_benchmark.ipynb     (L2, 30 types)
    python build_notebook.py l1         -> notebooks/marker_benchmark_l1.ipynb  (L1, 8 types)

The notebook contains only short orchestration calls into ``src`` (no heavy
logic). Figure/table names are suffixed with a notebook-level ``TAG`` so the two
granularities never clobber each other. Stability is loaded from the CLI-produced
cache at L2 (the 30-type bootstrap is slow) and computed inline at L1 (8 one-vs
-rest targets -> fast).
"""
import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

LEVELS = {
    "l1": {"level": "celltype.l1", "tag": "_l1", "name": "L1 (8 major lineages)",
           "n": "8", "inline_stability": True, "inline_tuning": True,
           "per_type_recovery": True, "figlevel": "l1",
           "file": "marker_benchmark_l1.ipynb"},
    "l2": {"level": "celltype.l2", "tag": "", "name": "L2 (~30 cell types)",
           "n": "30", "inline_stability": False, "inline_tuning": False,
           "per_type_recovery": False, "figlevel": "l2",
           "file": "marker_benchmark.ipynb"},
}


def build(cfg):
    C = []
    md = lambda t: C.append(("md", t))
    code = lambda t: C.append(("code", t))
    lvl, tag, name = cfg["level"], cfg["tag"], cfg["name"]
    figlevel = cfg["figlevel"]

    md(rf"""# Benchmarking dependency measures for marker-gene identification in multi-modal single-cell data

**PBMC CITE-seq (Hao et al. 2021) — a 2×2 decomposition of dependency measures.**
**Granularity: {name}.**

|            | Marginal                | Conditional                       |
|------------|-------------------------|-----------------------------------|
| **Linear**    | Spearman correlation    | Partial correlation (shrinkage)   |
| **Nonlinear** | Mutual information (KSG) | Integrated Gradients on an MLP     |

We ask whether *nonlinearity* and *multivariate context* change which genes are
identified as cell-type markers, and which axis matters more. Every method sees
**RNA only**; the surface-protein (ADT) modality is used solely to define a
cross-modal, protein-derived ground-truth driver set $D_c$ per cell type. Recovery
of $D_c$ is scored with the parameter-free $\mathrm{{AUC_{{rel}}}}$ metric.

All heavy logic lives in the importable `src/` package; this notebook only
orchestrates and narrates. See `README.md` for the full list of design decisions.""")

    code(f"""import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OpenMP runtime clash
if sys.platform == "win32":   # jupyter launches the kernel unactivated; expose env DLLs
    for _sub in ("Library/bin", "Library/mingw-w64/bin", "Library/usr/bin",
                 "Scripts", "bin", "DLLs"):
        _p = os.path.join(sys.prefix, _sub)
        if os.path.isdir(_p):
            os.add_dll_directory(_p)
            os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
import torch  # noqa: F401  import torch FIRST so its MKL/OpenMP DLLs win over the
              # conda stack's (otherwise torch's shm.dll fails with WinError 127)
sys.path.insert(0, "..")
import numpy as np, pandas as pd
from types import SimpleNamespace
from src import (config, data_io, preprocessing as pp, ground_truth as gt,
                 protein_gene_map as pgm, scorers, benchmark as bm, stats as st,
                 metric, plotting as P)
P.set_style()
P.set_fig_level("{figlevel}")   # figures -> results/figures/{figlevel}/{{pdf,png}}/
LEVEL = "{lvl}"      # annotation granularity benchmarked in this notebook
TAG = "{tag}"        # suffix so tables do not clobber other granularities
rng = np.random.default_rng(config.SEED)""")

    # --- Data ---
    md(r"""## 1. Data

The Hao 2021 3′ CITE-seq data (GEO `GSE164378`: whole-transcriptome RNA + 228
surface proteins + donor / `celltype.l1/l2/l3` labels) is downloaded and streamed
into a **sqrt-proportional stratified subsample** of ~25k cells by `src.data_io`
(one-time; cached). The same cached cells are used at every granularity.""")
    code("""mdata = data_io.build_dataset()      # uses on-disk cache if present
rna, adt = mdata["rna"], mdata["adt"]
print(f"{rna.n_obs:,} cells | RNA {rna.n_vars:,} genes | ADT {adt.n_vars} proteins")
rna.obs[LEVEL].value_counts()""")

    # --- Preprocessing ---
    md(r"""## 2. Preprocessing and the shared feature matrix

Standard Scanpy/muon workflow: QC, `log1p` for RNA, CLR for ADT, HVG selection.
The four methods then consume the **identical** matrix `X` — the
rank-transformed, z-scored expression of the shared gene universe
(**HVGs ∪ protein-encoding marker genes**). The per-gene average-rank transform
also collapses dropout zeros to a shared rank, mitigating the zero-inflation
confound for MI.""")
    code("""pp.compute_qc(rna)
pp.normalize_rna(rna)
pp.normalize_adt(adt)
hvg = pp.select_hvg(rna)
marker_genes = pgm.all_candidate_genes(adt.var_names) & set(rna.var_names)
universe = sorted(set(hvg) | marker_genes)
X, genes = pp.build_shared_matrix(rna, universe)
print(f"HVGs {len(hvg)} | marker genes {len(marker_genes)} | shared universe {len(genes)}")
print("shared matrix:", X.shape, X.dtype)""")
    code("""qc = rna.obs[["n_genes_by_counts","total_counts","pct_counts_mito"]].rename(
    columns={"n_genes_by_counts":"genes / cell","total_counts":"UMI / cell",
             "pct_counts_mito":"% mitochondrial"})
fig = P.qc_violins(qc, list(qc.columns)); P.save(fig, f"fig_qc{TAG}"); fig""")

    # --- EDA ---
    md(r"""## 3. Exploration: UMAP, donor batch effect, cell-type label validation

PCA → UMAP on the HVGs, with **Harmony** batch correction over the 8 donors. The
donor UMAPs (before/after) show whether batch correction is warranted.""")
    code("""pp.embed(rna, hvg, run_harmony=True)
CT = rna.obs[LEVEL].astype(str).values
donor = rna.obs[config.DONOR_KEY].astype(str).values""")
    code("""fig,_ = P.embedding_scatter(rna.obsm["X_umap"], CT); P.save(fig,f"fig_umap_celltype{TAG}"); fig""")
    code("""fig,_ = P.embedding_scatter(rna.obsm["X_umap"], donor); P.save(fig,f"fig_umap_donor{TAG}"); fig""")
    code("""fig,_ = P.embedding_scatter(rna.obsm["X_umap_harmony"], donor); P.save(fig,f"fig_umap_donor_harmony{TAG}"); fig""")
    md(r"""Cell-type labels validated against the **independent protein channel**: mean
CLR surface-protein level per cell type (z-scored per protein). The block-diagonal
structure (CD3→T, CD19/CD20→B, CD14→monocytes, CD56→NK, …) confirms the WNN labels
agree with direct molecular evidence.""")
    code("""key = [p for p in ["CD3-1","CD4-1","CD8","CD19","CD20","CD14","CD16","CD56-1",
                    "CD11c","HLA-DR","CD123","CD34","CD27","CD25"] if p in adt.var_names]
clr = pd.DataFrame(np.asarray(adt.layers["clr"]), columns=adt.var_names); clr["ct"]=CT
mat = clr.groupby("ct")[key].mean(); matz = (mat-mat.mean())/mat.std()
fig,_ = P.protein_validation_heatmap(matz); P.save(fig,f"fig_adt_validation{TAG}"); fig""")

    # --- Ground truth ---
    md(r"""## 4. Protein-derived ground truth $D_c$

For each cell type we take the top surface proteins by one-vs-rest Wilcoxon
*score* on CLR-ADT (p-values are uninformative at n≈25k), mapped to encoding
gene(s) by molecular fact only (`src.protein_gene_map`), intersected with the
scored gene universe. This is independent of all four RNA methods.""")
    code("""drivers, details = gt.build_ground_truth(adt, LEVEL, gene_universe=genes)
summary = gt.summarise_drivers(drivers)
summary.to_csv(config.tab_path(f"ground_truth_summary{TAG}"), index=False)
summary""")

    # --- Methods ---
    md(r"""## 5. The four dependency measures

All operate on the shared matrix `X` and a binary one-vs-rest indicator.""")
    md(r"""### 5.1 Linear · marginal — Spearman correlation
### 5.2 Linear · conditional — partial correlation (shrinkage)
The empirical gene covariance is ill-conditioned (dropout, collinearity, $p\approx n$),
so a naïve precision matrix is unstable. We use a **Ledoit–Wolf shrinkage**
covariance → precision → point-biserial partial correlation. The eigen-spectrum
below motivates the shrinkage.""")
    code("""from sklearn.covariance import LedoitWolf
Xs = X - X.mean(0, keepdims=True)
emp = np.linalg.eigvalsh(np.cov(Xs.T))
lw  = np.linalg.eigvalsh(LedoitWolf(assume_centered=True).fit(Xs).covariance_)
fig,_ = P.eigen_spectrum(emp, lw); P.save(fig,f"fig_pcorr_eigenspectrum{TAG}"); fig""")
    md(r"""### 5.3 Nonlinear · marginal — KSG mutual information
Kraskov–Stögbauer–Grassberger kNN estimator on the rank-transformed expression,
computed one-vs-rest per cell type (parallelised across cell types, estimated on a
fixed 10k-cell subsample for tractability).""")
    code("""valid = [ct for ct,d in drivers.items()
         if len([g for g in d if g in set(genes)]) >= config.MIN_DRIVERS]
mi_attr = bm.compute_mi_parallel(X, rna.obs[LEVEL].astype(str).values, valid,
                                 n_jobs=-1, seed=config.SEED)
_ex = valid[0]
fig,_ = P.score_hist(mi_attr[_ex], f"mutual information — {_ex}"); P.save(fig,f"fig_mi_hist{TAG}"); fig""")
    md(r"""### 5.4 Nonlinear · conditional — Integrated Gradients on an MLP
A small classifier with `tanh` hidden layers and a `softmax` output (predictions
on the simplex), trained with cross-entropy + inverse-frequency class weights and
early stopping on validation loss. Integrated Gradients then attributes each class
to the input genes.""")
    md(r"""**Architecture selection (CV macro-F1 sweep).** Overall accuracy hides
minority-class behaviour, so we select the `tanh`-MLP architecture by 3-fold
cross-validated **macro-F1** (every cell type weighted equally), which targets the
harder, less-abundant lineages. The winning configuration — hidden (512, 256),
dropout 0.4, lr 3e-4, weight decay 1e-4 — is the default in `src.mlp.train_mlp`.""")
    code("""from src import mlp as mlpmod
y = rna.obs[LEVEL].astype(str).values
default_cfg = dict(hidden=(256,128), dropout=0.2, lr=1e-3, weight_decay=1e-5)
tuned_cfg   = dict(hidden=(512,256), dropout=0.4, lr=3e-4, weight_decay=1e-4)""")
    if cfg["inline_tuning"]:
        md(r"""At this granularity (8 classes) the sweep is cheap, so it is run
inline. macro-F1 rises modestly overall but improves the harder classes most.""")
        code("""grid = [default_cfg,
        dict(hidden=(256,128), dropout=0.3, lr=3e-4, weight_decay=1e-4),
        dict(hidden=(512,256), dropout=0.3, lr=3e-4, weight_decay=1e-4),
        dict(hidden=(512,256,128), dropout=0.3, lr=3e-4, weight_decay=1e-4),
        dict(hidden=(256,128), dropout=0.3, lr=1e-4, weight_decay=1e-4),
        tuned_cfg]
sweep = mlpmod.hyperparameter_search(X, y, grid, n_splits=3, seed=config.SEED)
sweep.assign(hidden=sweep.hidden.astype(str)).to_csv(
    config.tab_path(f"mlp_hyperparam_search{TAG}"), index=False)
sweep""")
        code("""cvd = mlpmod.cross_validate(X, y, n_splits=3, seed=config.SEED, **default_cfg)
cvt = mlpmod.cross_validate(X, y, n_splits=3, seed=config.SEED, **tuned_cfg)
print(f"default macro-F1 {cvd['macro_f1']:.3f} | tuned macro-F1 {cvt['macro_f1']:.3f}")
f1 = {"default": mlpmod.per_class_f1(cvd["y_true"], cvd["y_pred"]),
      "tuned":   mlpmod.per_class_f1(cvt["y_true"], cvt["y_pred"])}
fig,_ = P.per_class_bar(f1); P.save(fig, f"fig_mlp_per_class_f1{TAG}"); fig""")
    else:
        md(r"""At this granularity (30 classes) the sweep is expensive, so it is run
via `run_pipeline.py tune` and its results / per-class diagnostics are loaded here.""")
        code("""from IPython.display import Image
sw = config.tab_path(f"mlp_hyperparam_search{TAG}")
display(pd.read_csv(sw)) if os.path.exists(sw) else None""")
        code("""# per-class F1 (default vs tuned) and confusion matrix from the CV sweep
for _n in ("fig_mlp_per_class_f1", "fig_mlp_confusion"):
    _f = P.fig_dir("png") / f"{_n}.png"
    if _f.exists(): display(Image(str(_f)))""")
    md(r"""The final classifier is trained on all cells with the tuned architecture
(the `src.mlp` default), then Integrated Gradients is computed.""")
    code("""trained = mlpmod.train_mlp(X, y, seed=config.SEED)
print(f"held-out test accuracy: {trained.test_acc:.3f}  (best epoch {trained.history.best_epoch})")
fig,_ = P.mlp_training_curves(trained.history); P.save(fig,f"fig_mlp_training{TAG}"); fig""")
    code("""attr = mlpmod.integrated_gradients(trained, X, y, seed=config.SEED)
classes = list(trained.classes)
ig_attr = {ct: attr[:, classes.index(ct)] for ct in classes}
print("IG attributions:", attr.shape, "(genes × classes)")""")
    md(r"""UMAP of the MLP's penultimate-layer representation — the nonlinear,
class-discriminative geometry the attributions are read from.""")
    code("""import umap
memb = mlpmod.mlp_embedding(trained, X)
mu = umap.UMAP(random_state=config.SEED).fit_transform(memb)
fig,_ = P.embedding_scatter(mu, CT); P.save(fig,f"fig_mlp_umap{TAG}"); fig""")

    # --- Benchmark ---
    md(r"""## 6. Benchmark: driver recovery ($\mathrm{AUC_{rel}}$)

$\mathrm{AUC_{rel}}$ is the normalized driver-recovery AUC (≡ Mann–Whitney ROC-AUC
of the per-gene score discriminating drivers from non-drivers): 0.5 = random,
1.0 = perfect, parameter-free, defined on every method's output.""")
    code("""store = {}
res = bm.run_benchmark(X, genes, y, drivers, mi_attr=mi_attr, ig_attr=ig_attr,
                       store_scores=store, seed=config.SEED)
res.to_csv(config.tab_path(f"benchmark_auc_rel{TAG}"), index=False)
res.pivot_table(index="celltype", columns="method", values="auc_rel").round(3)""")
    if cfg["per_type_recovery"]:
        md(r"""Cumulative recovery curves (x = fraction of all ranked genes), one
panel **per cell type** — recovery is strongly cell-type dependent: on clean
lineages Integrated Gradients can dominate the whole curve, on heterogeneous ones
it trails.""")
        code("""panel = {ct: {m: metric.recovery_curve(store[ct][m], store[ct]["_driver_mask"])
              for m in config.METHODS} for ct in sorted(store)}
fig,_ = P.recovery_curves_panel(panel, ncols=4); P.save(fig,f"fig_recovery_curves{TAG}"); fig""")
    else:
        md(r"""Cumulative recovery curve **averaged over cell types** (bootstrap-over
-cell-type band; x = fraction of all ranked genes). Its area is exactly
$\mathrm{AUC_{rel}}$.""")
        code("""panel = {ct: {m: metric.recovery_curve(store[ct][m], store[ct]["_driver_mask"])
              for m in config.METHODS} for ct in sorted(store)}
fig,_ = P.recovery_curve_mean(store, config.METHODS); P.save(fig,f"fig_recovery_curves{TAG}"); fig""")

    # --- Stats ---
    md(r"""## 7. Statistical comparison

The unit of replication is the **cell type** (paired across methods): a Friedman
omnibus test, then Holm-corrected pairwise Wilcoxon signed-rank tests with
matched-pairs rank-biserial effect sizes.""")
    code("""wide = st.pivot_auc(res)
fried = st.friedman_test(wide)
print(f"Friedman χ² = {fried['statistic']:.2f}, p = {fried['pvalue']:.2e} "
      f"(n = {fried['n_celltypes']} cell types)")
pw = st.pairwise_wilcoxon(wide, method_order=config.METHODS)
pw.to_csv(config.tab_path(f"pairwise_wilcoxon{TAG}"), index=False)
pw""")
    code("""sig = [(r.method_a, r.method_b, "*" if r["significant_0.05"] else "ns")
       for _, r in pw.iterrows()]
fig,_ = P.auc_box(res, config.METHODS, sig_pairs=sig); P.save(fig,f"fig_auc_box{TAG}"); fig""")
    code("""fig,_ = P.auc_heatmap(wide, method_order=config.METHODS); P.save(fig,f"fig_auc_heatmap{TAG}"); fig""")

    # --- Early recovery + sensitivity ---
    md(r"""### 7.1 Early recovery (recall@k) and sensitivity to heterogeneous categories

$\mathrm{AUC_{rel}}$ integrates the *entire* ranking, but a practitioner only
inspects the top handful of genes. **Recall@k** (fraction of drivers within the
top-k) targets that early regime; the band is a bootstrap over cell types (the
replication unit). Note the axes differ from the per-cell-type curves above: here
x is the **absolute** top-k (a few dozen genes) **averaged over all cell types**,
whereas those curves use the *fraction* of all genes for a *single* type — so a
single-type curve reaching 100% early and a middling cross-type average are
consistent, not contradictory. The whole-curve winner and the early-recovery
winner need not agree.""")
    if cfg["per_type_recovery"]:
        md(r"""Zoomed into the **early region** (top ~5% of ranked genes), the same
per-cell-type panels — this is the regime a practitioner actually inspects.""")
        code("""fig,_ = P.recovery_curves_panel(panel, ncols=4, xlim=(0, 0.05))
P.save(fig,f"fig_recovery_early{TAG}"); fig""")
    else:
        md(r"""**Recall@k averaged over cell types** (bootstrap-over-cell-type band);
x is the absolute top-k. The whole-curve winner and the early-recovery winner need
not agree, though here the marginal methods lead throughout.""")
        code("""rk = bm.recovery_at_k_table(store, methods=config.METHODS)
rk.to_csv(config.tab_path(f"recovery_at_k{TAG}"), index=False)
fig,_ = P.recovery_at_k_curve(rk, config.METHODS); P.save(fig,f"fig_recovery_at_k{TAG}"); fig""")
    md(r"""Heterogeneous grab-bag categories (e.g. `other`, `other T`) have weak,
ambiguous protein-derived ground truth and depress recovery for *all* methods —
Integrated Gradients most. The table below (mean AUC_rel with vs without them) is a
transparency check showing how much the aggregate understates the model-based
method on cleanly-defined cell types.""")
    code("""hetero = [c for c in res.celltype.unique() if "other" in c.lower()]
print("heterogeneous categories at this granularity:", hetero or "none")
sens = pd.DataFrame({"mean_auc_all": res.groupby("method").auc_rel.mean()})
if hetero:
    sens["mean_auc_excl_other"] = res[~res.celltype.isin(hetero)].groupby("method").auc_rel.mean()
sens.round(3).loc[config.METHODS]""")

    # --- Cross-method ---
    md(r"""## 8. Cross-method comparison (gene level)

Pooling every (gene, cell type) score (z-scored within cell type), how do the four
measures relate, and where do the protein-confirmed drivers sit? The pairwise
scatter matrix and the 3-D view (partial correlation × MI × IG) make the
agreement/disagreement structure explicit; protein-derived drivers are
highlighted.""")
    code("""long = bm.scores_to_long(store, genes)
long.to_parquet(config.DATA_PROC / f"scores_long{TAG}.parquet")
fig,_ = P.pairwise_scatter_matrix(long, config.METHODS); P.save(fig,f"fig_pairwise_scatter{TAG}"); fig""")
    code("""fig,_ = P.scatter3d(long, "partial_corr", "mi_ksg", "ig_mlp"); P.save(fig,f"fig_scatter3d{TAG}", tight=False); fig""")

    # --- Stability ---
    if cfg["inline_stability"]:
        md(r"""## 9. Stability and robustness

Descriptive stability bands (not inference): **bootstrap over cells** for the
closed-form methods, and **MLP-seed variation** for Integrated Gradients. At this
granularity the bootstrap is cheap, so it is computed inline.""")
        code("""boot = bm.bootstrap_over_cells(X, genes, y, drivers, n_boot=5,
                              boot_cells=3000, seed=config.SEED)
boot.to_csv(config.tab_path(f"stability_bootstrap{TAG}"))
boot.round(3)""")
        code("""seedf = bm.seed_stability_ig(X, genes, y, drivers, mi_attr,
                            seeds=(0,1,2), seed=config.SEED)
seedf.to_csv(config.tab_path(f"stability_seed{TAG}"), index=False)
seedf.round(3)""")
    else:
        md(r"""## 9. Stability and robustness

Descriptive stability bands (not inference): **bootstrap over cells** for the
methods, and **MLP-seed variation** for Integrated Gradients. At this granularity
the bootstrap recomputes MI / refits many shrinkage covariances, so it is produced
by `run_pipeline.py stability` / `seed` (CLI) and loaded here.""")
        code("""sb, ss = config.tab_path(f"stability_bootstrap{TAG}"), config.tab_path(f"stability_seed{TAG}")
if os.path.exists(sb): display(pd.read_csv(sb, index_col=0).round(3))
if os.path.exists(ss): display(pd.read_csv(ss).round(3))""")

    # --- Conclusion ---
    md(r"""## 10. Summary

The figures and tables above decompose marker-gene identification along the
linear/nonlinear and marginal/conditional axes against an independent,
protein-derived ground truth, at this annotation granularity. Comparing against
the other granularity's notebook indicates whether the conclusion (which
statistical assumption changes the recovered markers) is robust to how finely cell
types are defined. See the report for interpretation.""")
    return C


def write(cfg):
    nb = new_notebook()
    nb.cells = [new_markdown_cell(t) if k == "md" else new_code_cell(t)
                for k, t in build(cfg)]
    nb.metadata["kernelspec"] = {"name": "marker-bench",
                                 "display_name": "Python (marker-bench)",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    out = Path("notebooks") / cfg["file"]
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print("wrote", out, "with", len(nb.cells), "cells")


if __name__ == "__main__":
    key = sys.argv[1].lower() if len(sys.argv) > 1 else "l2"
    write(LEVELS[key])
