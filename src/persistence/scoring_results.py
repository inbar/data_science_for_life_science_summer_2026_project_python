from pathlib import Path

import pandas as pd

from src import config
from src.persistence import path_tools
import logging

log = logging.getLogger(__file__)

SCORING_DIR_NAME = "scores"
SUBDIR_TEMPLATE = "split_{split_size}/seed_{seed}"
SUBSAMPLED_DATA_SUBDIR_TEMPLATE = "from_subsampled_dataset/{subsample_size}"
FULL_DATASET_SUBDIR_NAME = "from_full_dataset"
INDEX_COLUMN_NAME = "gene_name"


def get_file_path(method_name: str,
                  subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                  level: str = config.DEFAULT_LEVEL,
                  test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                  seed: int = config.DEFAULT_SEED,
                  root_dir: Path = config.RESULTS_DIR_PATH,
                  tag: str = config.DEFAULT_TAG) -> Path:
    file_name = f"{method_name}_results.csv"
    subfolders = path_tools.get_subfolder_path(subsample_size=subsample_size,
                                               level=level,
                                               test_split_size=test_split_size,
                                               seed=seed,
                                               tag=tag)

    file_path = root_dir / "scores" / subfolders / file_name
    return file_path


def save_results(results: pd.DataFrame,
                 method: str,
                 subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                 level: str = config.DEFAULT_LEVEL,
                 test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                 seed: int = config.DEFAULT_SEED,
                 root_dir: Path = config.RESULTS_DIR_PATH,
                 tag: str = config.DEFAULT_TAG):
    file_path = get_file_path(method_name=method,
                              subsample_size=subsample_size,
                              level=level,
                              test_split_size=test_split_size,
                              seed=seed,
                              root_dir=root_dir,
                              tag=tag)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving results to file: {file_path}")
    results.to_csv(file_path,
                   index=True,
                   index_label=INDEX_COLUMN_NAME,
                   na_rep="NaN")


def load_results(method: str,
                 subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                 level: str = config.DEFAULT_LEVEL,
                 test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                 seed: int = config.DEFAULT_SEED,
                 root_dir: Path = config.RESULTS_DIR_PATH,
                 tag: str = config.DEFAULT_TAG) -> pd.DataFrame:
    file_path = get_file_path(method_name=method,
                              subsample_size=subsample_size,
                              level=level,
                              test_split_size=test_split_size,
                              seed=seed,
                              root_dir=root_dir,
                              tag=tag)
    return pd.read_csv(file_path, index_col=INDEX_COLUMN_NAME)


def load_multiple_results(methods: list[str],
                          subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                          level: str = config.DEFAULT_LEVEL,
                          test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                          seed: int = config.DEFAULT_SEED,
                          root_dir: Path = config.RESULTS_DIR_PATH,
                          tag: str = config.DEFAULT_TAG) -> dict[
    str, pd.DataFrame]:
    results_dict = {}
    for method in methods:
        results_dict[method] = load_results(method=method,
                                            subsample_size=subsample_size,
                                            level=level,
                                            test_split_size=test_split_size,
                                            seed=seed,
                                            root_dir=root_dir,
                                            tag=tag)
    return results_dict
