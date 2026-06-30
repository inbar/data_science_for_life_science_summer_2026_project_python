import pandas as pd
from anndata import AnnData
from torch import nn, optim

from src import config
from src.deep_learning import pytorch_device, data_conversion
from src.deep_learning.gene_expression_mlp_model import GeneExpressionModel
from src.preprocessing.rna import LAYER_NAME_SCALED

import logging

log = logging.getLogger(__file__)


def get_hyperparameters(model: nn.Module):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    return (
        criterion,
        optimizer
    )


def train(training_data: AnnData,
          labeling_df: pd.DataFrame,
          n_epochs=15,
          level: str = config.DEFAULT_LEVEL) -> GeneExpressionModel:
    device = pytorch_device.get_device()

    n_genes = training_data.n_vars
    n_cell_types = training_data.obs[level].nunique()

    log.info("Creating GeneExpressionModel")
    log.info("-----------------------------")
    log.info(f"   input_dim={n_genes} (n_genes)")
    log.info(f"   output_dim={n_cell_types} (n_cell_types")
    log.info("")
    model = GeneExpressionModel(n_genes, n_cell_types)

    log.info("Extracting the scaled data from the dataset...")
    training_dataset_scaled = training_data.to_df(layer=LAYER_NAME_SCALED)

    log.info("Fetching hyperparameters...")
    criterion, optimizer = get_hyperparameters(model)

    log.info("Creating dataset loader...")
    training_dataset, training_dataset_loader = data_conversion.to_dataset_loader(
        training_dataset_scaled,
        labeling_df)

    log.info("Starting MLP training loop...")
    log.info("-" * 40)

    for epoch in range(n_epochs):
        model.train()  # Explicitly set model to training mode (enables Dropout/BatchNorm)
        running_loss = 0.0

        for training_batch_x, training_batch_y in training_dataset_loader:
            # Move the batched chunk data to the same hardware device as the model
            training_batch_x, training_batch_y = training_batch_x.to(
                device), training_batch_y.to(device)

            # 1. Clear previous gradients
            optimizer.zero_grad()

            # 2. Forward Pass: Make predictions
            predictions = model(training_batch_x)

            # 3. Calculate error
            loss = criterion(predictions, training_batch_y)

            # 4. Backward Pass: Calculate adjustments
            loss.backward()

            # 5. Optimization Step: Update model weights
            optimizer.step()

            running_loss += loss.item() * training_batch_x.size(0)

        # Calculate average loss for this epoch
        epoch_loss = running_loss / len(training_dataset)
        log.info(
            f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {epoch_loss:.4f}")

    log.info("-" * 40)
    log.info("Training fit complete.")

    return model
