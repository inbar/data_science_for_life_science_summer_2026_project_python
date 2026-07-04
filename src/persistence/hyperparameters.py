import logging
from pathlib import Path

import yaml

from src import config

ROOT_DIR = config.PERSISTENCE_DIR

FILE_NAME = "best_params.yml"

log = logging.getLogger(__file__)


def get_file_path(root_dir: Path = ROOT_DIR) -> Path:
    file_path = root_dir / "hyperparameters" / FILE_NAME
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def save_best_params(best_params: dict[str, float],
                     root_dir: Path = ROOT_DIR):
    file_path = get_file_path(root_dir=root_dir)

    log.info(f"Saving best params to file: {file_path}")
    with open(file_path, "w") as f:
        yaml.dump(best_params, f)



def load_best_params(root_dir: Path = ROOT_DIR) -> dict:
    file_path = get_file_path(root_dir=root_dir)

    with open(file_path, "r") as f:
        file = yaml.safe_load(f)

    return file
