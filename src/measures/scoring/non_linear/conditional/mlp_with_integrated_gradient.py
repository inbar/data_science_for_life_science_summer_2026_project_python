import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

from src.deep_learning import data_conversion
from src.deep_learning.gene_expression_mlp_model import GeneExpressionModel
from src.deep_learning.pytorch_device import get_device

import logging

log = logging.getLogger(__file__)

def calculate_scores(trained_model: GeneExpressionModel,
                     expression_levels_df: pd.DataFrame,
                     labeling_df: pd.DataFrame) -> pd.DataFrame:
    """Computes importance scores for genes for each per cell type
        using Integrated Gradients (IG) on a trained model.

        For each cell type, the function isolates the relevant cells, computes
        how much each gene's expression contributed to the model's true
        prediction relative to a zero-expression baseline, and aggregates the
        attributions.

        Expectation from expression data:
            - normalized (log1p)? yes
            - Scaled? yes
            - Ranked? no

        Parameters
        ----------
        trained_model : GeneExpressionModel
            The trained PyTorch neural network model.
        expression_levels_df: A pandas DataFrame of shape (n_cells, n_genes)
            containing the gene expression matrix.
        labeling_df: A pandas DataFrame of shape (n_cells, n_cell_types)
            containing the true cell-type labels.

        Returns
        -------
        resutls: pd.DataFrame
            A DataFrame of shape (n_genes, n_cell_types) containing the
            mean integrated gradient attribution scores between each gene and cell type.
        """

    log.info("Computing Non-Linear/Conditional scoring: Integrated Gradient over a trained MLP")

    device = get_device()

    dataset_tensor, _ = data_conversion.to_dataset_loader(
        expression_levels_df, labeling_df)

    X_tensor, Y_tensor = dataset_tensor.tensors

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

        # Take the absolute value of the attributions.
        # A negative attribution is an indicator the same way as
        # a positive one.
        raw_attributions = attributions.detach().cpu().numpy()
        abs_attributions = np.abs(raw_attributions)
        mean_attributions = abs_attributions.mean(axis=0)
        all_attributions[cell_type] = mean_attributions

    log.info("-" * 50)
    log.info("IG Computation Complete.")

    gene_names = expression_levels_df.columns

    mlp_ig_results_df = pd.DataFrame(
        all_attributions,
        index=gene_names
    )

    return mlp_ig_results_df