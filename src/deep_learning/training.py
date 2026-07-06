from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from torch import nn, optim

from src import config
from src.deep_learning import pytorch_device, data_conversion
from src.deep_learning.gene_expression_mlp_model import GeneExpressionModel
from src.preprocessing.rna import LAYER_NAME_SCALED

import logging

log = logging.getLogger(__file__)


@dataclass
class TrainHistory:
    """Per-epoch training diagnostics (for the training-curve plot)."""
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    best_epoch: int = 0


def _eval_loss_acc(model, loader, loss_criterion, device):
    model.eval()
    total_loss, n, correct, total = 0.0, 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_criterion(logits, y).item() * x.size(0)
            n += x.size(0)
            correct += (torch.argmax(logits, 1) == torch.argmax(y, 1)).sum().item()
            total += x.size(0)
    return total_loss / max(n, 1), correct / max(total, 1)


def train(model: GeneExpressionModel,
          training_data: AnnData,
          labeling_df: pd.DataFrame,
          n_epochs=15,
          learning_rate: float = 1e-3,
          weight_decay: float = 1e-4,
          batch_size: int = 64,
          validation_data: AnnData = None,
          labeling_df_val: pd.DataFrame = None) -> tuple[GeneExpressionModel, TrainHistory]:
    """Train the MLP on the normalized + z-scored (``LAYER_NAME_SCALED``) matrix.

    Returns ``(model, history)``. If ``validation_data`` is given, per-epoch
    validation loss/accuracy are tracked and ``history.best_epoch`` marks the epoch
    with the lowest validation loss (the weights of that epoch are restored).
    """
    device = pytorch_device.get_device()
    model.to(device)

    log.info("Extracting the scaled data from the dataset...")
    training_dataset_scaled = training_data.to_df(layer=LAYER_NAME_SCALED)

    loss_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)

    training_dataset_tensor, training_dataset_loader = data_conversion.to_dataset_loader(
        training_dataset_scaled, labeling_df, batch_size=batch_size)

    val_loader = None
    if validation_data is not None and labeling_df_val is not None:
        _, val_loader = data_conversion.to_dataset_loader(
            validation_data.to_df(layer=LAYER_NAME_SCALED), labeling_df_val,
            batch_size=batch_size)

    history = TrainHistory()
    best_val = float("inf")
    best_state = None

    log.info("Starting MLP training loop...")
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for training_batch_x, training_batch_y in training_dataset_loader:
            training_batch_x = training_batch_x.to(device)
            training_batch_y = training_batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(training_batch_x)
            loss = loss_criterion(predictions, training_batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * training_batch_x.size(0)

        epoch_loss = running_loss / len(training_dataset_tensor)
        history.train_loss.append(epoch_loss)

        if val_loader is not None:
            v_loss, v_acc = _eval_loss_acc(model, val_loader, loss_criterion, device)
            history.val_loss.append(v_loss)
            history.val_acc.append(v_acc)
            if v_loss < best_val:
                best_val, history.best_epoch = v_loss, epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            log.info(f"Epoch {epoch + 1}/{n_epochs} | train {epoch_loss:.4f} "
                     f"| val {v_loss:.4f} | val_acc {v_acc:.3f}")
        else:
            log.info(f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {epoch_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    log.info("Training complete.")
    return model, history


def evaluate(model: GeneExpressionModel,
             data: AnnData,
             labeling_df: pd.DataFrame,
             batch_size: int = 256):
    """Predicted vs. true class indices on ``data`` (for confusion / per-class F1).

    Returns ``(y_true, y_pred, classes)`` where classes are the ``labeling_df``
    columns in order.
    """
    device = pytorch_device.get_device()
    model.eval()
    X = data.to_df(layer=LAYER_NAME_SCALED)
    Y = labeling_df.loc[X.index.values]
    x = torch.tensor(X.values, dtype=torch.float32)
    y_true = np.argmax(Y.values, axis=1)
    preds = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            logits = model(x[i:i + batch_size].to(device))
            preds.append(torch.argmax(logits, 1).cpu().numpy())
    return y_true, np.concatenate(preds), list(labeling_df.columns)
