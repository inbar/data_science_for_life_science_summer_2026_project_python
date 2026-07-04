import pandas as pd
from anndata import AnnData
from torch import nn, optim

from src import config
from src.deep_learning import pytorch_device, data_conversion
from src.deep_learning.gene_expression_mlp_model import GeneExpressionModel
from src.preprocessing.rna import LAYER_NAME_SCALED

import logging

log = logging.getLogger(__file__)


def get_hyperparameters(model: nn.Module,
                        learning_rate: float = 1e-3,
                        weight_decay: float = 1e-4):
    loss_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay)

    return (
        loss_criterion,
        optimizer
    )


def train(model: GeneExpressionModel,
          training_data: AnnData,
          labeling_df: pd.DataFrame,
          n_epochs=15,
          learning_rate: float = 1e-3,
          weight_decay: float = 1e-4,
          batch_size: int = 64) -> GeneExpressionModel:
    device = pytorch_device.get_device()
    model.to(device)

    log.info("Extracting the scaled data from the dataset...")
    training_dataset_scaled = training_data.to_df(layer=LAYER_NAME_SCALED)

    log.info("Fetching hyperparameters...")
    loss_criterion, optimizer = get_hyperparameters(model,
                                                    learning_rate,
                                                    weight_decay)

    log.info("Creating dataset loader...")
    training_dataset_tensor, training_dataset_loader = data_conversion.to_dataset_loader(
        training_dataset_scaled,
        labeling_df,
        batch_size=batch_size
    )

    log.info("Starting MLP training loop...")
    log.info("-" * 40)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for training_batch_x, training_batch_y in training_dataset_loader:
            training_batch_x = training_batch_x.to(device)
            training_batch_y = training_batch_y.to(device)

            # 1. Clear previous gradients
            optimizer.zero_grad()

            # 2. Make predictions
            predictions = model(training_batch_x)

            # 3. Calculate error
            loss = loss_criterion(predictions, training_batch_y)

            # 4. Calculate adjustments
            loss.backward()

            # 5. Update model weights
            optimizer.step()

            running_loss += loss.item() * training_batch_x.size(0)

        # Calculate average loss for this epoch
        epoch_loss = running_loss / len(training_dataset_tensor)
        log.info(
            f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {epoch_loss:.4f}")

    log.info("-" * 40)
    log.info("Training complete.")

    return model
