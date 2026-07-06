import optuna
import pandas as pd
import torch
from anndata import AnnData
from optuna import Study
from sklearn.metrics import f1_score

from src import config
from src.deep_learning import pytorch_device
from src.deep_learning import data_conversion
from src.deep_learning import training
from src.deep_learning.gene_expression_mlp_model import GeneExpressionModel
from src.preprocessing.rna import LAYER_NAME_SCALED


def get_objective_function(training_data: AnnData,
                           test_data: AnnData,
                           labeling_df_training: pd.DataFrame,
                           labeling_df_test: pd.DataFrame,
                           seed: int = config.DEFAULT_SEED):
    def objective(trial):
        # The hyperparameter matrix
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3,
                                            log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        input_dropout = trial.suggest_float("input_dropout", 0.1, 0.5)

        # Deterministic per-trial seed: reproducible given (seed, n_trials), but
        # each trial still gets a fresh (not identical) init/shuffle draw, so no
        # single hyperparameter combination is favoured or penalised purely by
        # reusing the same init across every trial.
        torch.manual_seed(seed + trial.number)

        n_genes = training_data.n_vars
        n_celltypes = len(labeling_df_training.columns)
        model = GeneExpressionModel(
            input_dim=n_genes,
            output_dim=n_celltypes,
            dropout_rate=input_dropout)

        # Train the model
        training.train(model=model,
                       training_data=training_data,
                       labeling_df=labeling_df_training,
                       n_epochs=15,
                       learning_rate=learning_rate,
                       weight_decay=weight_decay,
                       batch_size=batch_size)

        # Evaluate performance using the Macro F1-score
        model.eval()
        all_predictions = []
        all_targets = []

        _, test_dataset_loader = data_conversion.to_dataset_loader(
            test_data.to_df(LAYER_NAME_SCALED),
            labeling_df_test,
            batch_size=batch_size
        )

        device = pytorch_device.get_device()
        with torch.no_grad():
            for batch_x, batch_y in test_dataset_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                predictions = torch.argmax(logits, dim=1)
                targets = torch.argmax(batch_y, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.numpy())

        macro_f1 = f1_score(all_targets, all_predictions, average="macro")

        return macro_f1

    return objective


def tune(training_data: AnnData,
         test_data: AnnData,
         labeling_df_training: pd.DataFrame,
         labeling_df_test: pd.DataFrame,
         n_trials:int = config.N_TRIALS,
         seed: int = config.DEFAULT_SEED) -> Study:
    # NOTE: previously unseeded -- which hyperparameters got tried (TPESampler's
    # own search order) and each trial's model init/shuffling were both governed
    # by ambient RNG state, so re-running tuning with "the same" --seed could
    # (and did) land on a different best_params.yml every time.
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))

    objective_function = get_objective_function(
        training_data=training_data,
        test_data=test_data,
        labeling_df_training=labeling_df_training,
        labeling_df_test=labeling_df_test,
        seed=seed
    )

    study.optimize(objective_function, n_trials=n_trials)

    return study
