import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def to_dataset_loader(dataset: pd.DataFrame,
                      labeling_df: pd.DataFrame,
                      batch_size=64):
    X = dataset.values
    Y = labeling_df.loc[dataset.index.values].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    dataset_tensor = TensorDataset(X_tensor, Y_tensor)
    dataset_loader = DataLoader(dataset_tensor,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True)

    return dataset_tensor, dataset_loader
