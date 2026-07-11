from datetime import datetime
import logging
import os
import shutil
from pathlib import Path

import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from src import config

ROOT_DIR = config.PERSISTENCE_DIR

FILE_NAME = "best_params.yml"

HISTORY_FILE_NAME = "optuna_study_history.log"

log = logging.getLogger(__file__)


def get_file_path(root_dir: Path = ROOT_DIR) -> Path:
    file_path = root_dir / "parameter_tuning" / FILE_NAME
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path

def get_optuna_storage(root_dir: Path = ROOT_DIR, delete_if_exists = False):
    file_path = root_dir / "parameter_tuning" / HISTORY_FILE_NAME

    original = Path(file_path)

    if delete_if_exists and original.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{original.stem}_{timestamp}{original.suffix}"
        backup_path = original.parent / backup_name
        shutil.copy2(original, backup_path)
        os.remove(file_path)

    return JournalStorage(JournalFileBackend(str(file_path)))

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

