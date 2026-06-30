from pathlib import Path

import pandas as pd

from src import config
from src.persistence import path_tools
import logging

log = logging.getLogger(__file__)


ROOT_DIR = config.RESULTS_DIR_PATH / "scores"
SUBDIR_TEMPLATE = "split_{split_size}/seed_{seed}"
SUBSAMPLED_DATA_SUBDIR_TEMPLATE = "from_subsampled_dataset/{subsample_size}"
FULL_DATASET_SUBDIR_NAME = "from_full_dataset"
INDEX_COLUMN_NAME = "gene_name"

def get_file_path(method_name: str,
                  subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                  level: str = config.DEFAULT_LEVEL,
                  test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                  seed: int = config.DEFAULT_SEED) -> Path:

    file_name = f"{method_name}_results.csv"
    subfolders = path_tools.get_subfolder_path(subsample_size=subsample_size,
                                               level=level,
                                               test_split_size=test_split_size,
                                               seed=seed)

    file_path = ROOT_DIR / subfolders / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def save_results(results: pd.DataFrame,
                 method_name: str,
                 subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                 level: str = config.DEFAULT_LEVEL,
                 test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                 seed: int = config.DEFAULT_SEED):
    file_path = get_file_path(method_name=method_name,
                              subsample_size=subsample_size,
                              level=level,
                              test_split_size=test_split_size,
                              seed=seed)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(file_path,
                   index=True,
                   index_label=INDEX_COLUMN_NAME,
                   na_rep="NaN")


def load_results(method_name: str,
                 subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                 level: str = config.DEFAULT_LEVEL,
                 test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                 seed: int = config.DEFAULT_SEED) -> pd.DataFrame:
    file_path = get_file_path(method_name=method_name,
                              subsample_size=subsample_size,
                              level=level,
                              test_split_size=test_split_size,
                              seed=seed)
    return pd.read_csv(file_path, index_col=INDEX_COLUMN_NAME)
