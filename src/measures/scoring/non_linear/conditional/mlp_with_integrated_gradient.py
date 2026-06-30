import pandas as pd
import torch
from captum.attr import IntegratedGradients

from src.deep_learning import data_conversion
from src.deep_learning.gene_expression_mlp_model import GeneExpressionModel
from src.deep_learning.pytorch_device import get_device

from src.logs import get_logger

log = get_logger()

def calculate_scores(trained_model: GeneExpressionModel,
                     expression_levels_df: pd.DataFrame,
                     labeling_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing Non-Linear/Conditional scoring: Integrated Gradient over a trained MLP")

    device = get_device()

    tensor_dataset, dataset_loader = data_conversion.to_dataset_loader(
        expression_levels_df, labeling_df)

    X_tensor, Y_tensor = tensor_dataset.tensors

    # Initialize the Integrated Gradients tool the trained model
    ig = IntegratedGradients(trained_model)

    # Empty dictionary to store the attributions
    all_attributions = {}

    log.info("Computing Integrated Gradients across all cell types...")
    log.info("-" * 40)

    for cell_type_index, cell_type in enumerate(labeling_df.columns):
        log.info(f"Processing attributions for: {cell_type}")

        # We are only interested in the results for cells of type cell_type.
        # We don't need to calculate the attributions for the rest.
        non_zero_cells = (Y_tensor[:, cell_type_index] == 1)
        X_subset = X_tensor[non_zero_cells].to(device)
        baseline = torch.zeros_like(X_subset).to(device)

        attributions: torch.Tensor = ig.attribute(
            inputs=X_subset,
            baselines=baseline,
            target=cell_type_index,
            n_steps=50,
            # Minimal internal_batch_size = Number of cells (rows)
            internal_batch_size=X_subset.shape[0]
        )

        mean_attributions = attributions.detach().cpu().numpy().mean(axis=0)
        all_attributions[cell_type] = mean_attributions

    log.info("-" * 50)
    log.info("IG Computation Complete.")

    gene_names = expression_levels_df.columns

    mlp_ig_results_df = pd.DataFrame(
        all_attributions,
        index=gene_names
    )

    return mlp_ig_results_df