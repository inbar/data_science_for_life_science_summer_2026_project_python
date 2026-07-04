from torch import nn

from src.deep_learning.pytorch_device import get_device
from src.persistence.models import load_trained_model_weights


# TODO:
# - Tapered dropout
# - Activation: consider Mish/GELU or LeakyReLU
class GeneExpressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(GeneExpressionModel, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )

        self.hidden = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(dropout_rate / 2),  # Tapered dropout

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(0.1)
        )

        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden(x)
        return self.output(x)


def load_trained_model(n_genes: int,
                       n_cells: int,
                       test_split_size: int,
                       seed: int,
                       subsample_size: int) -> GeneExpressionModel:
    state_dict = load_trained_model_weights(test_split_size, seed,
                                            subsample_size)

    model = GeneExpressionModel(n_genes, n_cells)

    model.load_state_dict(state_dict)
    model.to(get_device())
    return model
