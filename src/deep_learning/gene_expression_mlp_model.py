from torch import nn

from deep_learning.pytorch_device import get_device
from persistence.models import load_trained_model


class GeneExpressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(GeneExpressionModel, self).__init__()

        # A robust 3-layer architecture for scaling single-cell data
        self.network = nn.Sequential(
            # Layer 1: Input (Number of Genes) -> 512 hidden nodes
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(dropout_rate),

            # Layer 2: 512 -> 256 hidden nodes
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(dropout_rate),

            # Layer 3: 256 -> Output (Your 30 Cell Types)
            # We output raw logits (no sigmoid here, BCEWithLogitsLoss handles it safely)
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def load_trained_model(n_genes: int,
                       n_cells: int,
                       test_split_size_pct: int,
                       seed: int,
                       subsample_size: int = None) -> GeneExpressionModel:
    state_dict = load_trained_model(test_split_size_pct, seed, subsample_size)

    model = GeneExpressionModel(n_genes, n_cells)

    model.load_state_dict(state_dict)
    model.to(get_device())
    return model
