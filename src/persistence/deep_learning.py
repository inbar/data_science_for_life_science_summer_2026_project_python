import torch
from torch import nn

from config import ROOT

TRAINED_MODEL_FILE_NAME_TEMPATE = "model_weights_split_{test_split_size_pct}_seed_{seed}.pt"
DATASET_FILENAME_TEMPLATE = "multi_modal_dataset__{modifier}.h5mu"
TRAINED_MODELS_DIR_PATH = ROOT / "persistence/models/trained_models/"


def get_trained_model_name(test_split_size_pct: int,
                           seed: int) -> str:
    return DATASET_FILENAME_TEMPLATE.format(split=test_split_size_pct, seed=seed)


def save_trained_model_weights(model: nn.Module,
                               test_split_size_pct: int,
                               seed: int):

    file_name = get_trained_model_name(test_split_size_pct, seed)
    torch.save(model.state_dict(), TRAINED_MODELS_DIR_PATH / file_name)


def load_trained_model_weights(test_split_size_pct: int,
                               seed: int):

    file_name = get_trained_model_name(test_split_size_pct, seed)
    return torch.load(TRAINED_MODELS_DIR_PATH / file_name)

