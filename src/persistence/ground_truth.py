from pathlib import Path
import yaml

from src import config
from src.persistence import path_tools

ROOT_DIR = config.PERSISTENCE_DIR

FILE_NAME = "ground_truth.yml"


def get_file_path(root_dir: Path = ROOT_DIR,
                  test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                  seed: int = config.DEFAULT_SEED,
                  subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                  level: str = config.DEFAULT_LEVEL) -> Path:
    subfolders = path_tools.get_subfolder_path(subsample_size=subsample_size,
                                               level=level,
                                               test_split_size=test_split_size,
                                               seed=seed)

    file_path = root_dir / "ground_truth" / subfolders / FILE_NAME
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def save_ground_truth(ground_truth: dict[str, set],
                      root_dir: Path = ROOT_DIR,
                      test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                      seed: int = config.DEFAULT_SEED,
                      subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                      level: str = config.DEFAULT_LEVEL):
    file_path = get_file_path(root_dir=root_dir,
                              test_split_size=test_split_size,
                              seed=seed,
                              subsample_size=subsample_size,
                              level=level)

    with open(file_path, "w") as f:
        yaml.dump(ground_truth, f)


def load_ground_truth(root_dir: Path = ROOT_DIR,
                      test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                      seed: int = config.DEFAULT_SEED,
                      subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                      level: str = config.DEFAULT_LEVEL) -> dict[str, set]:
    file_path = get_file_path(root_dir=root_dir,
                              test_split_size=test_split_size,
                              seed=seed,
                              subsample_size=subsample_size,
                              level=level)

    with open(file_path, "r") as f:
        ground_truth = yaml.safe_load(f)

    return ground_truth
