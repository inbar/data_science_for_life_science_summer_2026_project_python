import logging
from pathlib import Path

from mudata import MuData

from src import config
from src.persistence import datasets
from src.persistence import path_tools

log = logging.getLogger(__file__)

ROOT_DIR = config.PERSISTENCE_DIR / "data_splits"
SPLIT_DIR_NAME_TEMPLATE = "split_{test_split_size}/seed_{seed}"

SUBSAMPLED_DATA_SUBDIR_TEMPLATE = "from_subsampled_dataset/{subsample_size}"
FULL_DATASET_SUBDIR_NAME = "from_full_dataset"

TRAINING_FILE_NAME = "training.h5mu"
TEST_FILE_NAME = "test.h5mu"

DEFAULT_SPLIT_NAME = "initial_split"
HVG_SPLIT_NAME = "hvg_split"


def get_base_path(split_name: str = DEFAULT_SPLIT_NAME,
                  subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                  level: str = config.DEFAULT_LEVEL,
                  test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                  seed: int = config.DEFAULT_SEED) -> Path:
    subfolders = path_tools.get_subfolder_path(subsample_size=subsample_size,
                                               level=level,
                                               test_split_size=test_split_size,
                                               seed=seed)

    return ROOT_DIR / subfolders / split_name


def get_training_file_path(split_name: str = DEFAULT_SPLIT_NAME,
                           subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                           level: str = config.DEFAULT_LEVEL,
                           test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                           seed: int = config.DEFAULT_SEED) -> Path:
    base_path = get_base_path(split_name=split_name,
                              subsample_size=subsample_size,
                              level=level,
                              test_split_size=test_split_size,
                              seed=seed)

    return base_path / TRAINING_FILE_NAME


def get_test_file_path(split_name: str = DEFAULT_SPLIT_NAME,
                       subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                       level: str = config.DEFAULT_LEVEL,
                       test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                       seed: int = config.DEFAULT_SEED) -> Path:
    base_path = get_base_path(split_name=split_name,
                              subsample_size=subsample_size,
                              level=level,
                              test_split_size=test_split_size,
                              seed=seed)

    return base_path / TEST_FILE_NAME


def save_split(training_data: MuData,
               test_data: MuData,
               split_name: str = DEFAULT_SPLIT_NAME,
               test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
               seed: int = config.DEFAULT_SEED,
               subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
               level: str = config.DEFAULT_LEVEL):
    training_split_size_pct = 100 - test_split_size

    log.info(f"Persist data split to disk:")
    log.info(f"  split_name: {split_name}")
    log.info(f"  training/test: "
             f"{test_split_size}/{training_split_size_pct}")
    log.info(f"  seed: {seed}")
    log.info(f"  subsample_size: {subsample_size}")
    log.info(f"  level: {level}")

    training_file = get_training_file_path(split_name=split_name,
                                           subsample_size=subsample_size,
                                           level=level,
                                           test_split_size=test_split_size,
                                           seed=seed)

    test_file = get_test_file_path(split_name=split_name,
                                   subsample_size=subsample_size,
                                   level=level,
                                   test_split_size=test_split_size,
                                   seed=seed)

    log.info(f"Training data file: {training_file}")
    datasets.save_mudata_dataset_to_disk(training_data, training_file)

    log.info(f"Test data file: {test_file}")
    datasets.save_mudata_dataset_to_disk(test_data, test_file)


def load_training_data(split_name: str = DEFAULT_SPLIT_NAME,
                       test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                       seed: int = config.DEFAULT_SEED,
                       subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                       level: str = config.DEFAULT_LEVEL) -> MuData:
    training_file = get_training_file_path(split_name=split_name,
                                           subsample_size=subsample_size,
                                           level=level,
                                           test_split_size=test_split_size,
                                           seed=seed)

    if not training_file.exists():
        log.error(
            f"Split data does not exist (test_split_size: {test_split_size}, seed: {seed})")
        log.error(
            "Check the split size/seed are correct or create the split first.")

    return datasets.read_h5mu_file(training_file)


def load_test_data(split_name: str = DEFAULT_SPLIT_NAME,
                   test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                   seed: int = config.DEFAULT_SEED,
                   subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                   level: str = config.DEFAULT_LEVEL) -> MuData:
    test_file = get_test_file_path(split_name=split_name,
                                   subsample_size=subsample_size,
                                   level=level,
                                   test_split_size=test_split_size,
                                   seed=seed)

    if not test_file.exists():
        log.error(
            f"Split data does not exist (test_split_size: {test_split_size}, seed: {seed})")
        log.error(
            "Check the split size/seed are correct or create the split first.")

    return datasets.read_h5mu_file(test_file)


def load_split_data(split_name: str = DEFAULT_SPLIT_NAME,
                    test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                    seed: int = config.DEFAULT_SEED,
                    subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                    level: str = config.DEFAULT_LEVEL) -> tuple[MuData, MuData]:
    return (
        load_training_data(split_name,
                           test_split_size,
                           seed,
                           subsample_size,
                           level),
        load_test_data(split_name,
                       test_split_size,
                       seed,
                       subsample_size,
                       level)
    )
