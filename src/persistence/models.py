from pathlib import Path

import torch
from torch import nn

from src.persistence import path_tools
from src import config

FILE_NAME = "trained_model_weights.pt"

ROOT_DIR = config.PERSISTENCE_DIR / "trained_models"


def get_trained_model_file_path(
    test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
    seed: int = config.DEFAULT_SEED,
    subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
    level: str = config.DEFAULT_LEVEL) -> Path:
    subfolders = path_tools.get_subfolder_path(subsample_size=subsample_size,
                                               level=level,
                                               test_split_size=test_split_size,
                                               seed=seed)

    file_path = ROOT_DIR / subfolders / FILE_NAME
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def save_trained_model_weights(model: nn.Module,
                               test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                               seed: int = config.DEFAULT_SEED,
                               subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                               level: str = config.DEFAULT_LEVEL):
    file_path = get_trained_model_file_path(test_split_size,
                                            seed,
                                            subsample_size,
                                            level)
    torch.save(model.state_dict(), file_path)


def load_trained_model_weights(
    test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
    seed: int = config.DEFAULT_SEED,
    subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
    level: str = config.DEFAULT_LEVEL):
    file_path = get_trained_model_file_path(test_split_size,
                                            seed,
                                            subsample_size,
                                            level)
    return torch.load(file_path)
