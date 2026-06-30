import pandas as pd
from anndata import AnnData
from torch import nn, optim

from deep_learning import pytorch_device, data_conversion
from deep_learning.gene_expression_mlp_model import GeneExpressionModel
from logs import get_logger

log = get_logger()


def get_hyperparameters(model: nn.Module):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    return (
        criterion,
        optimizer
    )


def train(training_dataset: AnnData,
          labeling_df: pd.DataFrame,
          n_epochs=15) -> GeneExpressionModel:
    device = pytorch_device.get_device()

    n_genes = len(training_dataset.var)
    n_cells = len(labeling_df.columns)

    log.info(
        f"Creating GeneExpressionModel with (input_dim={n_genes} (n_genes)), (output_dim={n_cells} (n_cells)) ")

    model = GeneExpressionModel(n_genes, n_cells)

    # TODO: validate: the training data must be scaled!
    criterion, optimizer = get_hyperparameters(model)
    training_dataset, training_dataset_loader = data_conversion.to_dataset_loader(
        training_dataset.to_df(),
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
