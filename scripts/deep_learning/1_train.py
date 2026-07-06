#!/usr/bin/env python

# To run the scripts run:
# source setup_environment.sh
# In the project root

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import umap
from anndata import ImplicitModificationWarning
from pandas.errors import PerformanceWarning
from sklearn.metrics import f1_score, confusion_matrix

from src.deep_learning.gene_expression_mlp_model import GeneExpressionModel
from src import config
from src import logs
from src.deep_learning import training
from src.deep_learning.pytorch_device import get_device
from src.persistence import parameter_tuning as hyperparam_persistence
from src.persistence import models as model_persistence
from src.persistence import path_tools
from src.persistence import splits as split_persistence
from src.preprocessing import rna as rna_preprocessing

warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

import logging

logs.setup_logging(__file__)
log = logging.getLogger(__file__)


def get_diagnostics_dir(root_dir, subsample_size, level, test_split_size, seed, tag):
    subfolders = path_tools.get_subfolder_path(subsample_size=subsample_size,
                                               level=level,
                                               test_split_size=test_split_size,
                                               seed=seed, tag=tag)
    diagnostics_dir = root_dir / "mlp_diagnostics" / subfolders
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    return diagnostics_dir


def main(args):
    subsample_size = args.subsample_size
    level = args.level
    test_split_size = args.test_split_size
    seed = args.seed
    n_epochs = args.n_epochs
    tag = args.tag
    root_dir = args.root_dir

    log.info(f"Running Training")
    log.info("===================")
    for k, v in vars(parsed_args).items():
        log.info(f"   {k}: {v}")
    log.info("")

    log.info("Loading split data...")
    log.info("")
    # Both splits are loaded (unlike the original version of this script, which
    # only loaded training data): validation-based early stopping needs a held-out
    # set, and it is the intended design (see README: "early stopping on
    # validation loss prevents the small net from over-fitting").
    training_data, test_data = split_persistence.load_split_data(
        split_name=split_persistence.HVG_SPLIT_NAME,
        test_split_size=test_split_size,
        subsample_size=subsample_size,
        seed=seed,
        level=level
    )
    training_data_rna = training_data["rna"]
    test_data_rna = test_data["rna"]
    target_df_training = rna_preprocessing.build_target_df(training_data_rna, level)
    target_df_test = rna_preprocessing.build_target_df(test_data_rna, level)

    # training.train() reads the "scaled" layer -- must be computed here or it
    # fails with a KeyError (this layer is not persisted by earlier stages).
    rna_preprocessing.apply_scaling_to_split_data(training_data_rna, test_data_rna)

    log.info("Loading hyperparameter values from file...")
    hyperparams = hyperparam_persistence.load_best_params(
        root_dir=config.LOCAL_DATA_ROOT)

    n_genes = training_data_rna.n_vars
    n_celltypes = training_data_rna.obs[level].nunique()

    log.info("Creating new model instance...")
    model = GeneExpressionModel(
        input_dim=n_genes,
        output_dim=n_celltypes,
        dropout_rate=hyperparams["input_dropout"]
    )

    trained_model, history = training.train(model=model,
                                            training_data=training_data_rna,
                                            labeling_df=target_df_training,
                                            n_epochs=n_epochs,
                                            learning_rate=hyperparams["learning_rate"],
                                            weight_decay=hyperparams["weight_decay"],
                                            batch_size=hyperparams["batch_size"],
                                            validation_data=test_data_rna,
                                            labeling_df_val=target_df_test)
    log.info(f"Training done | best_epoch {history.best_epoch} | "
             f"final val_acc {history.val_acc[-1]:.3f}")

    model_persistence.save_trained_model_weights(trained_model,
                                                 test_split_size=test_split_size,
                                                 seed=seed,
                                                 subsample_size=subsample_size,
                                                 level=level,
                                                 tag=tag)

    # MLP diagnostics (training curves, per-class F1 / confusion, penultimate-layer
    # embedding), repo-local so the report notebook can load them.
    diagnostics_dir = get_diagnostics_dir(root_dir, subsample_size, level,
                                         test_split_size, seed, tag)
    json.dump({"train_loss": history.train_loss, "val_loss": history.val_loss,
               "val_acc": history.val_acc, "best_epoch": history.best_epoch},
              open(diagnostics_dir / "history.json", "w"), indent=1)

    y_true, y_pred, classes = training.evaluate(trained_model, test_data_rna,
                                                target_df_test)
    np.savez(diagnostics_dir / "eval.npz", y_true=y_true, y_pred=y_pred,
             classes=np.array(classes))
    log.info(f"test accuracy {np.mean(y_true == y_pred):.3f} | macro-F1 "
             f"{f1_score(y_true, y_pred, average='macro'):.3f}")

    trained_model.eval()
    with torch.no_grad():
        embedding = trained_model.embed(torch.tensor(
            test_data_rna.to_df(rna_preprocessing.LAYER_NAME_SCALED).values,
            dtype=torch.float32).to(get_device())).cpu().numpy()
    mlp_umap = umap.UMAP(random_state=seed).fit_transform(embedding)
    np.savez(diagnostics_dir / "umap.npz", coords=mlp_umap,
             celltype=test_data_rna.obs[level].astype(str).values)

    log.info(f"Done. MLP diagnostics written to {diagnostics_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_size", type=int,
                        default=config.DEFAULT_SUBSAMPLE_SIZE)
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--test_split_size", type=int,
                        default=config.DEFAULE_TEST_SPLIT_SIZE)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--n_epochs", type=int,
                        default=config.DEFAULT_N_EPOCHS)
    parser.add_argument("--tag", type=str, default=config.DEFAULT_TAG)
    parser.add_argument("--root_dir", type=str, default=str(config.LOCAL_DATA_ROOT),
                        help="Repo-local root for MLP diagnostics (the model "
                             "weights themselves still go to the out-of-repo "
                             "persistence cache via model_persistence).")
    parsed_args = parser.parse_args()
    parsed_args.root_dir = Path(parsed_args.root_dir)

    main(parsed_args)
