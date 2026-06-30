from pathlib import Path

import torch
from torch import nn

from src import config

TRAINED_MODELS_DIR_PATH = config.PERSISTANCE_DIR / "trained_models"
SUBSAMPLED_DATA_SUBDIR_TEMPLATE = "from_subsampled_dataset/{subsample_size}"
FULL_DATASET_SUBDIR_NAME = "from_full_dataset"
TRAINED_MODEL_SUBDIR_TEMPLATE = "split_{split_size}/seed_{seed}"

FILE_NAME = "trained_model_weights.pt"


def get_trained_model_file_path(
    test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
    seed: int = config.DEFAULT_SEED,
    subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
    level: str = config.DEFAULT_LEVEL) -> Path:
    if subsample_size is None:
        subsample_dir = FULL_DATASET_SUBDIR_NAME
    else:
        subsample_dir = SUBSAMPLED_DATA_SUBDIR_TEMPLATE.format(
            subsample_size=subsample_size)

    trained_model_subdir = TRAINED_MODEL_SUBDIR_TEMPLATE.format(
        subsample_size=subsample_size,
        split_size=test_split_size,
        seed=seed
    )

    full_file_path = TRAINED_MODELS_DIR_PATH / level / subsample_dir / trained_model_subdir / FILE_NAME
    full_file_path.parent.mkdir(parents=True, exist_ok=True)

    return full_file_path


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
