#!/usr/bin/env python
"""Validation run of the full pipeline on celltype.l1 (8 types) + a 10k-cell
subsample. This is the *computation* half: it runs every heavy stage once and
persists the artifacts, so the report notebook (pipeline_validation_l1_10k.ipynb)
only has to load and plot them.

Run:  python scripts/validate_pipeline_l1_10k.py
Artifacts are written under  local_data/validation_l1_10k/.
"""
import os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import mudata as md
import torch
import umap
from sklearn.model_selection import train_test_split

from src import config
from src import ground_truth as gt, mappings
from src.preprocessing import rna as rna_pp, adt as adt_pp, normalization as norm_pp
from src.exploratory_analysis import dim_reduction as dr
from src.deep_learning.gene_expression_mlp_model import GeneExpressionModel
from src.deep_learning import training as mlp_training
from src.deep_learning.pytorch_device import get_device
from src.measures.scoring.linear.marginal import spearman_correlation
from src.measures.scoring.linear.conditional import ledoit_wolf_partial_correlation
from src.measures.scoring.non_linear.marginal import mutual_information_ksg
from src.measures.scoring.non_linear.conditional import mlp_with_integrated_gradient
from src.persistence import scoring_results as scoring_persistence
from src.persistence import ground_truth as gt_persistence

sc.settings.verbosity = 1

# --- validation configuration -----------------------------------------------
LEVEL = "celltype.l1"
SEED = config.DEFAULT_SEED
N_SUB = 10_000
TEST_SPLIT_SIZE = 40           # % test
N_EPOCHS = 20
RANK, SCALED = rna_pp.LAYER_NAME_RANK_TRANSFORMED, rna_pp.LAYER_NAME_SCALED

# Raw cells come from my_pipeline (Inbar's repo has only score CSVs). Override with
# the VALIDATION_H5MU env var if the data lives elsewhere.
H5MU = os.environ.get(
    "VALIDATION_H5MU",
    r"c:\Users\kr3ss\Desktop\data_science_project\my_pipeline\data\processed\pbmc3p_citeseq.h5mu")

VAL_ROOT = config.LOCAL_DATA_ROOT / "validation_l1_10k"
MLP_DIR = VAL_ROOT / "mlp"
EXP_DIR = VAL_ROOT / "exploratory"
for _d in (MLP_DIR, EXP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# canonical surface-protein markers to sanity-check the L1 lineages
ADT_VALIDATION_PROTEINS = ["CD3-1", "CD4-1", "CD8", "CD19", "CD20", "CD14", "CD16",
                           "CD56-1", "CD11c", "HLA-DR", "CD123", "CD34"]


def _dense(x):
    return np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def expr_df(dataset, layer):
    return pd.DataFrame(
        dataset.to_df(layer).values, index=dataset.obs_names,
        columns=pd.Index(dataset.var["gene_name"].values, name="gene_name"))


def main():
    t0 = time.time()

    # 1. Load + stratified subsample -----------------------------------------
    log(f"loading {H5MU}")
    mdata = md.read_h5mu(H5MU)
    rna, adt = mdata["rna"].copy(), mdata["adt"].copy()
    idx, _ = train_test_split(np.arange(rna.n_obs), train_size=N_SUB,
                              stratify=rna.obs[LEVEL].values, random_state=SEED)
    rna, adt = rna[idx].copy(), adt[idx].copy()
    log(f"subsample {rna.n_obs} cells | {rna.obs[LEVEL].nunique()} {LEVEL} types")

    # 2. Preprocess ----------------------------------------------------------
    rna.var["gene_name"] = rna.var_names.astype(str)
    adt.var["protein_name"] = adt.var_names.astype(str)
    rna_pp.calculate_qc_metrics_in_place(rna)
    norm_pp.normalize_in_place(rna)
    rna_pp.annotate_highly_variable_genes(rna)
    adt.obs[LEVEL] = adt.obs[LEVEL].astype("category")
    adt_pp.normalize_in_place(adt)

    # 3. Shared universe -----------------------------------------------------
    hvg = set(rna.var_names[rna.var["highly_variable"]])
    markers = mappings.get_marker_genes_for_proteins(adt.var_names) & set(rna.var_names)
    universe = sorted(hvg | markers)
    rna_u = rna[:, universe].copy()
    rna_u.var["gene_name"] = rna_u.var_names.astype(str)
    log(f"universe {len(universe)} (HVG {len(hvg)} + markers {len(markers)})")

    # 3b. Exploratory artifacts (QC, UMAP +/- Harmony, ADT label validation) --
    qc = rna.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].rename(
        columns={"n_genes_by_counts": "genes / cell", "total_counts": "UMI / cell",
                 "pct_counts_mt": "% mitochondrial"})
    qc.to_csv(EXP_DIR / "qc.csv", index=False)

    rna_h = rna[:, sorted(hvg)].copy()
    sc.pp.scale(rna_h, max_value=10)
    dr.perform_pca_in_place(rna_h)
    dr.perform_umap_in_place(rna_h)
    dr.perform_pca_harmony_in_place(rna_h)
    dr.perform_umap_harmony_in_place(rna_h)
    np.savez(EXP_DIR / "embeddings.npz",
             umap=rna_h.obsm[dr.OBSM_NAME_UMAP],
             umap_harmony=rna_h.obsm[dr.OBSM_NAME_UMAP_HARMONY],
             celltype=rna.obs[LEVEL].astype(str).values,
             donor=rna.obs[config.DONOR_KEY].astype(str).values)

    key = [p for p in ADT_VALIDATION_PROTEINS if p in adt.var_names]
    clr_df = pd.DataFrame(_dense(adt.layers[adt_pp.LAYER_NAME_CENTERED_LOG_RATIO]),
                          columns=adt.var_names)
    clr_df["ct"] = rna.obs[LEVEL].astype(str).values
    means = clr_df.groupby("ct")[key].mean()
    ((means - means.mean()) / means.std()).to_csv(EXP_DIR / "adt_validation.csv")
    log("exploratory artifacts saved (qc, embeddings, adt_validation)")

    # 4. Split + shared feature layers ---------------------------------------
    tr_idx, te_idx = train_test_split(
        np.arange(rna_u.n_obs), test_size=TEST_SPLIT_SIZE / 100,
        stratify=rna_u.obs[LEVEL].values, random_state=SEED)
    train, test = rna_u[tr_idx].copy(), rna_u[te_idx].copy()
    rna_pp.apply_scaling_to_split_data(train, test)
    rna_pp.apply_rank_transform_to_split_data(train, test)
    y_train = rna_pp.build_target_df(train, LEVEL)
    y_test = rna_pp.build_target_df(test, LEVEL)
    log(f"train {train.n_obs} | test {test.n_obs}")

    # 5. Ground truth (CLR) + persist ----------------------------------------
    drivers, gt_details = gt.build_ground_truth(
        adt, genes_of_interest=universe, level=LEVEL,
        layer=adt_pp.LAYER_NAME_CENTERED_LOG_RATIO)
    gt_persistence.save_ground_truth(
        drivers, root_dir=VAL_ROOT, level=LEVEL, subsample_size=N_SUB,
        test_split_size=TEST_SPLIT_SIZE, seed=SEED)
    log("ground truth drivers/type:", {k: len(v) for k, v in drivers.items()})

    # 6. Train MLP (normalized + z-scored) -----------------------------------
    torch.manual_seed(SEED)
    model = GeneExpressionModel(input_dim=len(universe),
                                output_dim=y_train.shape[1], dropout_rate=0.3)
    model, history = mlp_training.train(
        model, train, y_train, n_epochs=N_EPOCHS,
        validation_data=test, labeling_df_val=y_test)
    log(f"MLP done | best_epoch {history.best_epoch} | val_acc {history.val_acc[-1]:.3f}")

    # 7. Score the four methods + persist ------------------------------------
    def save(method, results):
        results.index.name = scoring_persistence.INDEX_COLUMN_NAME
        scoring_persistence.save_results(
            results=results, method=method, subsample_size=N_SUB, level=LEVEL,
            test_split_size=TEST_SPLIT_SIZE, seed=SEED, root_dir=VAL_ROOT)

    log("scoring: Spearman")
    save(config.METHOD_SPEARMAN,
         spearman_correlation.calculate_scores(expr_df(test, RANK), y_test))
    log("scoring: partial correlation")
    save(config.METHOD_PARTIAL_CORRELATION,
         ledoit_wolf_partial_correlation.calculate_scores(expr_df(test, RANK), y_test))
    log("scoring: MI (Ross 2014)")
    save(config.METHOD_MI_KSG,
         mutual_information_ksg.calculate_scores(
             expr_df(test, SCALED), y_test,
             k_neighbors=config.DEFAULT_K_NEIGHBORS, seed=SEED))
    log("scoring: Integrated Gradients (MLP)")
    ig = mlp_with_integrated_gradient.calculate_scores(model, expr_df(test, SCALED), y_test)
    save(config.METHOD_IG_MLP, ig.abs())   # IG importance = |attribution|

    # 8. MLP diagnostics -> plain files --------------------------------------
    json.dump({"train_loss": history.train_loss, "val_loss": history.val_loss,
               "val_acc": history.val_acc, "best_epoch": history.best_epoch},
              open(MLP_DIR / "history.json", "w"), indent=1)
    y_true, y_pred, classes = mlp_training.evaluate(model, test, y_test)
    np.savez(MLP_DIR / "eval.npz", y_true=y_true, y_pred=y_pred, classes=np.array(classes))
    model.eval()
    with torch.no_grad():
        emb = model.embed(torch.tensor(test.to_df(SCALED).values,
                                       dtype=torch.float32).to(get_device())).cpu().numpy()
    mlp_umap = umap.UMAP(random_state=SEED).fit_transform(emb)
    np.savez(MLP_DIR / "umap.npz", coords=mlp_umap,
             celltype=test.obs[LEVEL].astype(str).values)

    # metadata the report needs
    json.dump({"level": LEVEL, "subsample_size": N_SUB,
               "test_split_size": TEST_SPLIT_SIZE, "seed": SEED,
               "n_universe": len(universe), "n_train": int(train.n_obs),
               "n_test": int(test.n_obs), "methods": config.METHODS},
              open(VAL_ROOT / "run_metadata.json", "w"), indent=1)

    log(f"DONE in {time.time() - t0:.0f}s -> artifacts in {VAL_ROOT}")


if __name__ == "__main__":
    main()
