"""End-to-end driver used to validate the whole pipeline on the real persistence before
freezing the steps into the notebook. Caches expensive intermediates so reruns
are cheap. Run stages with: python run_pipeline.py <stage>
where stage in {prep, gt, mlp, bench, stats, all}.
"""
import os, sys, time, pickle, faulthandler
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OpenMP runtime clash
faulthandler.enable()
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
# NOTE: torch (src.mlp) is imported lazily inside stage_mlp only, so its OpenMP
# runtime does not clash with numba/MKL during the UMAP step of stage_prep.
from src import (config, persistence, preprocessing as pp, ground_truth as gt,
                 benchmark as bm, stats as st)

PROC = config.PROCESSED_DATA_DIR_PATH
LEVEL = config.PRIMARY_LEVEL


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def stage_prep():
    log("loading cache")
    md = persistence.load_dataset()
    rna, adt = md["rna"], md["adt"]
    log("RNA", rna.shape, "ADT", adt.shape)
    pp.compute_qc(rna)
    pp.normalize_rna(rna)
    pp.normalize_adt(adt)
    hvg = pp.select_hvg(rna)
    # Shared gene universe = HVGs U all protein-encoding marker genes present in
    # the persistence, so every protein-derived driver gene is actually scoreable (no
    # leakage: features remain RNA-only; ADT only labels which genes are "true").
    from src import mappings as pgm
    marker_genes = pgm.get_marker_genes_for_proteins(adt.var_names) & set(rna.var_names)
    universe = sorted(set(hvg) | marker_genes)
    log("HVGs", len(hvg), "| marker genes present", len(marker_genes),
        "| shared universe", len(universe))
    X, genes = pp.build_shared_matrix(rna, universe)
    log("shared matrix", X.shape, X.dtype)
    # --- save core artifacts BEFORE the (fragile, native) embedding step ---
    np.save(PROC / "X_ranked.npy", X)
    with open(PROC / "prep.pkl", "wb") as f:
        pickle.dump({"genes": genes, "hvg": hvg,
                     "obs": rna.obs, "var": rna.var}, f)
    adt.write(PROC / "adt_prep.h5ad")
    log("core prep cached; starting embeddings")
    pp.embed(rna, hvg, run_harmony=True)
    log("embeddings done", list(rna.obsm))
    np.savez(PROC / "embeddings.npz",
             **{k: rna.obsm[k] for k in rna.obsm if k.startswith("X_")})
    log("prep cached")


def stage_embed():
    md = persistence.load_dataset()
    rna = md["rna"]
    pp.normalize_rna(rna)
    with open(PROC / "prep.pkl", "rb") as f:
        hvg = pickle.load(f)["hvg"]
    log("computing embeddings (PCA/UMAP/Harmony)")
    pp.embed(rna, hvg, run_harmony=True)
    log("embeddings done", list(rna.obsm))
    np.savez(PROC / "embeddings.npz",
             **{k: rna.obsm[k] for k in rna.obsm if k.startswith("X_")})
    log("embeddings cached")


def stage_gt():
    import anndata as ad
    adt = ad.read_h5ad(PROC / "adt_prep.h5ad")
    with open(PROC / "prep.pkl", "rb") as f:
        prep = pickle.load(f)
    drivers, details = gt.build_ground_truth(adt, LEVEL, genes_of_interest=prep["genes"])
    summ = gt.pretty_print_ground_truth(drivers)
    log("ground truth:\n" + summ.to_string())
    details.to_csv(config.tab_path("ground_truth_proteins"), index=False)
    summ.to_csv(config.tab_path("ground_truth_summary"), index=False)
    with open(PROC / "drivers.pkl", "wb") as f:
        pickle.dump(drivers, f)


def stage_mlp():
    from src import mlp as mlpmod  # lazy: torch only loaded here
    X = np.load(PROC / "X_ranked.npy")
    with open(PROC / "prep.pkl", "rb") as f:
        prep = pickle.load(f)
    y = prep["obs"][LEVEL].astype(str).values
    log("training MLP on", X.shape, "classes", len(set(y)))
    t = mlpmod.train_mlp(X, y, seed=config.SEED, verbose=False)
    log(f"MLP test acc {t.test_acc:.3f}, best epoch {t.history.best_epoch}")
    attr = mlpmod.integrated_gradients(t, X, y, seed=config.SEED)
    log("IG attr", attr.shape)
    emb = mlpmod.mlp_embedding(t, X)
    np.save(PROC / "ig_attr.npy", attr)
    np.save(PROC / "mlp_emb.npy", emb)
    with open(PROC / "mlp.pkl", "wb") as f:
        pickle.dump({"classes": np.asarray(t.classes),
                     "history": {"train_loss": t.history.train_loss,
                                 "val_loss": t.history.val_loss,
                                 "val_acc": t.history.val_acc,
                                 "best_epoch": t.history.best_epoch},
                     "test_acc": t.test_acc}, f)


def stage_bench():
    X = np.load(PROC / "X_ranked.npy")
    attr = np.load(PROC / "ig_attr.npy")
    with open(PROC / "prep.pkl", "rb") as f:
        prep = pickle.load(f)
    with open(PROC / "drivers.pkl", "rb") as f:
        drivers = pickle.load(f)
    with open(PROC / "mlp.pkl", "rb") as f:
        mlpres = pickle.load(f)
    genes = prep["genes"]
    y = prep["obs"][LEVEL].astype(str).values
    classes = list(mlpres["classes"])
    ig_attr = {ct: attr[:, classes.index(ct)] for ct in classes}
    valid = [ct for ct, d in drivers.items()
             if len([g for g in d if g in set(genes)]) >= config.MIN_DRIVERS]
    mi_cache = PROC / "mi_attr.pkl"
    if mi_cache.exists():
        log("loading cached MI")
        mi_attr = pickle.load(open(mi_cache, "rb"))
    else:
        log(f"computing MI in parallel for {len(valid)} cell types")
        mi_attr = bm.compute_mi_parallel(X, y, valid, n_jobs=-1, seed=config.SEED)
        with open(mi_cache, "wb") as f:
            pickle.dump(mi_attr, f)
    store = {}
    log("running benchmark")
    res = bm.run_benchmark(X, genes, y, drivers, mi_attr=mi_attr, ig_attr=ig_attr,
                           store_scores=store, seed=config.SEED)
    res.to_csv(config.tab_path("benchmark_auc_rel"), index=False)
    long = bm.scores_to_long(store, genes)
    long.to_parquet(PROC / "scores_long.parquet")
    log("benchmark:\n" + res.pivot_table(index="celltype", columns="method",
                                          values="auc_rel").round(3).to_string())


def stage_stats():
    res = pd.read_csv(config.tab_path("benchmark_auc_rel"))
    wide = st.pivot_auc(res)
    fr = st.friedman_test(wide)
    log("Friedman:", fr)
    pw = st.pairwise_wilcoxon(wide, method_order=config.METHODS)
    log("pairwise:\n" + pw.to_string())
    pw.to_csv(config.tab_path("pairwise_wilcoxon"), index=False)


def stage_plots():
    import anndata as ad
    import matplotlib; matplotlib.use("Agg")
    import umap
    from types import SimpleNamespace
    from src import plotting as P
    P.set_style()
    P.set_fig_level("l2")   # run_pipeline operates on the L2 (primary) level

    with open(PROC / "prep.pkl", "rb") as f:
        prep = pickle.load(f)
    obs, genes = prep["obs"], prep["genes"]
    emb = np.load(PROC / "embeddings.npz")
    drivers = pickle.load(open(PROC / "drivers.pkl", "rb"))
    res = pd.read_csv(config.tab_path("benchmark_auc_rel"))
    long = pd.read_parquet(PROC / "scores_long.parquet")
    mlpres = pickle.load(open(PROC / "mlp.pkl", "rb"))
    adt = ad.read_h5ad(PROC / "adt_prep.h5ad")

    L1 = obs["celltype.l1"].astype(str).values
    L2 = obs[LEVEL].astype(str).values
    donor = obs[config.DONOR_KEY].astype(str).values

    # ---- EDA ----
    qc = obs[["n_genes_by_counts", "total_counts", "pct_counts_mito"]].rename(
        columns={"n_genes_by_counts": "genes / cell",
                 "total_counts": "UMI counts / cell",
                 "pct_counts_mito": "% mitochondrial"})
    P.save(P.qc_violins(qc, list(qc.columns)), "fig_qc")
    f, _ = P.embedding_scatter(emb["X_umap"], L1, legend_ncol=1); P.save(f, "fig_umap_celltype_l1")
    f, _ = P.embedding_scatter(emb["X_umap"], donor, legend_ncol=1); P.save(f, "fig_umap_donor")
    f, _ = P.embedding_scatter(emb["X_umap_harmony"], donor, legend_ncol=1); P.save(f, "fig_umap_donor_harmony")
    f, _ = P.embedding_scatter(emb["X_umap_harmony"], L1, legend_ncol=1); P.save(f, "fig_umap_celltype_harmony")

    # ADT label validation: mean CLR per L1 cell type for canonical proteins
    key_prot = ["CD3-1", "CD4-1", "CD8", "CD19", "CD20", "CD14", "CD16", "CD56-1",
                "CD11c", "HLA-DR", "CD123", "CD34", "CD27", "CD25"]
    key_prot = [p for p in key_prot if p in adt.var_names]
    clr = pd.DataFrame(np.asarray(adt.layers["clr"]), columns=adt.var_names)
    clr["ct"] = L1
    mat = clr.groupby("ct")[key_prot].mean()
    matz = (mat - mat.mean()) / mat.std()
    f, _ = P.protein_marker_validation_heatmap(matz); P.save(f, "fig_adt_validation")

    # ---- per-method diagnostics ----
    hist = SimpleNamespace(**mlpres["history"])
    f, _ = P.mlp_training_curves(hist); P.save(f, "fig_mlp_training")
    log("computing MLP-embedding UMAP")
    memb = np.load(PROC / "mlp_emb.npy")
    mu = umap.UMAP(random_state=config.SEED).fit_transform(memb)
    f, _ = P.embedding_scatter(mu, L1); P.save(f, "fig_mlp_umap")

    # partial-correlation justification: covariance eigen-spectrum raw vs shrinkage
    from sklearn.covariance import LedoitWolf
    X = np.load(PROC / "X_ranked.npy")
    Xs = X - X.mean(0, keepdims=True)
    emp = np.linalg.eigvalsh(np.cov(Xs.T))
    lw = np.linalg.eigvalsh(LedoitWolf(assume_centered=True).fit(Xs).covariance_)
    f, _ = P.eigen_spectrum(emp, lw); P.save(f, "fig_pcorr_eigenspectrum")

    # example cell type: MI histogram + recovery curves
    ex = "CD14 Mono"
    exl = long[long.celltype == ex]
    f, _ = P.score_hist(exl["mi_ksg"], "mutual information (z)"); P.save(f, "fig_mi_hist")
    from src.metric import recovery_curve
    curves = {m: recovery_curve(exl[m].values, exl["is_driver"].values)
              for m in config.METHODS if m in exl}
    f, _ = P.recovery_curves(curves); P.save(f, "fig_recovery_curves")

    # ---- results ----
    pw = pd.read_csv(config.tab_path("pairwise_wilcoxon")) if \
        config.tab_path("pairwise_wilcoxon").exists() else None
    sig = None
    if pw is not None:
        sig = [(r.method_a, r.method_b, "*" if r["significant_0.05"] else "ns")
               for _, r in pw.iterrows()]
    f, _ = P.auc_box(res, config.METHODS, sig_pairs=sig); P.save(f, "fig_auc_box")
    wide = res.pivot_table(index="celltype", columns="method", values="auc_rel")
    f, _ = P.auc_heatmap(wide, method_order=config.METHODS); P.save(f, "fig_auc_heatmap")

    # ---- cross-method (gene-level) ----
    f, _ = P.pairwise_scatter_matrix(long, config.METHODS); P.save(f, "fig_pairwise_scatter")
    f, _ = P.scatter3d(long, "partial_corr", "mi_ksg", "ig_mlp"); P.save(f, "fig_scatter3d", tight=False)
    log("all figures written to", config.FIGURES_DIR_PATH)


def stage_stability(n_boot: int = 5, boot_cells: int = 3000):
    """Bootstrap over cells (closed-form methods) -> AUC_rel CIs.

    Each iteration resamples ``boot_cells`` cells with replacement (a subsample;
    enough to characterise resampling variability) and recomputes the three
    closed-form scorers. The MLP/IG method's stability is reported separately via
    seed variation (stage_seed), since bootstrapping it would require retraining.
    """
    X = np.load(PROC / "X_ranked.npy")
    with open(PROC / "prep.pkl", "rb") as f:
        prep = pickle.load(f)
    drivers = pickle.load(open(PROC / "drivers.pkl", "rb"))
    genes = prep["genes"]
    y = prep["obs"][LEVEL].astype(str).values
    log(f"bootstrapping closed-form methods ({n_boot}x, {boot_cells} cells)")
    summ = bm.bootstrap_over_cells(X, genes, y, drivers, n_boot=n_boot,
                                   boot_cells=boot_cells, seed=config.SEED)
    log("bootstrap summary:\n" + summ.to_string())
    summ.to_csv(config.tab_path("stability_bootstrap"))


def stage_seed():
    """MLP/IG seed variation -> AUC_rel stability for ig_mlp."""
    X = np.load(PROC / "X_ranked.npy")
    with open(PROC / "prep.pkl", "rb") as f:
        prep = pickle.load(f)
    drivers = pickle.load(open(PROC / "drivers.pkl", "rb"))
    mi_attr = pickle.load(open(PROC / "mi_attr.pkl", "rb"))  # MI independent of seed
    genes = prep["genes"]
    y = prep["obs"][LEVEL].astype(str).values
    df = bm.seed_stability_ig(X, genes, y, drivers, mi_attr, seeds=(0, 1, 2),
                              seed=config.SEED)
    log("seed stability:\n" + df.to_string())
    df.to_csv(config.tab_path("stability_seed"), index=False)


def stage_tune():
    """Hyperparameter sweep for the MLP (ranked by CV macro-F1) + per-class
    diagnostics. macro-F1 rewards getting minority lineages right, directly
    targeting the classes Integrated Gradients recovers poorly."""
    import matplotlib; matplotlib.use("Agg")
    from sklearn.metrics import confusion_matrix
    from src import mlp as mlpmod, plotting as P
    P.set_style()
    P.set_fig_level("l2")   # L2 (primary) hyperparameter diagnostics
    X = np.load(PROC / "X_ranked.npy")
    with open(PROC / "prep.pkl", "rb") as f:
        prep = pickle.load(f)
    y = prep["obs"][LEVEL].astype(str).values

    default_cfg = dict(hidden=(256, 128), dropout=0.2, lr=1e-3, weight_decay=1e-5)
    grid = [
        default_cfg,
        dict(hidden=(256, 128),      dropout=0.3, lr=3e-4, weight_decay=1e-4),
        dict(hidden=(512, 256),      dropout=0.3, lr=3e-4, weight_decay=1e-4),
        dict(hidden=(512, 256, 128), dropout=0.3, lr=3e-4, weight_decay=1e-4),
        dict(hidden=(256, 128),      dropout=0.3, lr=1e-4, weight_decay=1e-4),
        dict(hidden=(512, 256),      dropout=0.4, lr=3e-4, weight_decay=1e-4),
    ]
    log(f"hyperparameter search ({len(grid)} configs x 3-fold CV)")
    res = mlpmod.hyperparameter_search(X, y, grid, n_splits=3, seed=config.SEED)
    res_save = res.copy(); res_save["hidden"] = res_save["hidden"].astype(str)
    res_save.to_csv(config.tab_path("mlp_hyperparam_search"), index=False)
    log("search results:\n" + res_save.to_string())

    best = res.iloc[0]
    best_cfg = dict(hidden=best["hidden"], dropout=float(best["dropout"]),
                    lr=float(best["lr"]), weight_decay=float(best["weight_decay"]))
    log(f"best config: {best_cfg}")

    cvd = mlpmod.cross_validate(X, y, n_splits=3, seed=config.SEED, **default_cfg)
    cvb = mlpmod.cross_validate(X, y, n_splits=3, seed=config.SEED, **best_cfg)
    f1d = mlpmod.per_class_f1(cvd["y_true"], cvd["y_pred"])
    f1b = mlpmod.per_class_f1(cvb["y_true"], cvb["y_pred"])
    P.save(P.per_class_bar({"default": f1d, "tuned": f1b})[0],
           "fig_mlp_per_class_f1")
    cls = sorted(set(y))
    cm = confusion_matrix(cvb["y_true"], cvb["y_pred"], labels=cls)
    P.save(P.confusion_heatmap(cm, cls)[0], "fig_mlp_confusion")
    log(f"default  macro-F1 {cvd['macro_f1']:.3f}  acc {cvd['acc']:.3f}")
    log(f"tuned    macro-F1 {cvb['macro_f1']:.3f}  acc {cvb['acc']:.3f}")
    pd.DataFrame({"default": f1d, "tuned": f1b}).to_csv(
        config.tab_path("mlp_per_class_f1"))


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    core = ["prep", "gt", "mlp", "bench", "stats"]
    stages = core if args == ["all"] else args
    for s in stages:
        log(f"=== STAGE {s} ===")
        globals()[f"stage_{s}"]()
    log("PIPELINE DONE")
