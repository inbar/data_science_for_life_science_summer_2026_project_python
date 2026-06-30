from pathlib import Path

from mudata import MuData

from src import config
from src.persistence import datasets

import logging

log = logging.getLogger(__file__)

SPLITS_ROOT_DIR = config.PERSISTANCE_DIR / "data_splits"
SPLIT_DIR_NAME_TEMPLATE = "split_{test_split_size}/seed_{seed}"

SUBSAMPLED_DATA_SUBDIR_TEMPLATE = "from_subsampled_dataset/{subsample_size}"
FULL_DATASET_SUBDIR_NAME = "from_full_dataset"

TRAINING_FILE_NAME = "training.h5mu"
TEST_FILE_NAME = "test.h5mu"

DEFAULT_SPLIT_NAME = "initial_split"
HVG_SPLIT_NAME = "hvg_split"


def get_split_dir_path(test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                       seed: int = config.DEFAULT_SEED,
                       subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                       level: str = config.DEFAULT_LEVEL) -> Path:
    if subsample_size is config.DEFAULT_SUBSAMPLE_SIZE:
        subsample_dir = FULL_DATASET_SUBDIR_NAME
    else:
        subsample_dir = SUBSAMPLED_DATA_SUBDIR_TEMPLATE.format(
            subsample_size=subsample_size)

    split_dir_name = SPLIT_DIR_NAME_TEMPLATE.format(
        training_split_size=100 - test_split_size,
        test_split_size=test_split_size,
        seed=seed)

    return SPLITS_ROOT_DIR / level / subsample_dir / split_dir_name


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

    split_dir_path = get_split_dir_path(test_split_size,
                                        seed,
                                        subsample_size,
                                        level)
    training_file = split_dir_path / split_name / TRAINING_FILE_NAME
    test_file = split_dir_path / split_name / TEST_FILE_NAME

    log.info(f"Training data file: {training_file}")
    datasets.save_mudata_dataset_to_disk(training_data, training_file)

    log.info(f"Test data file: {test_file}")
    datasets.save_mudata_dataset_to_disk(test_data, test_file)


def load_training_data(split_name: str = DEFAULT_SPLIT_NAME,
                       test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                       seed: int = config.DEFAULT_SEED,
                       subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                       level: str = config.DEFAULT_LEVEL) -> MuData:
    split_dir_path = get_split_dir_path(test_split_size,
                                        seed,
                                        subsample_size,
                                        level)
    training_file = split_dir_path / split_name / TRAINING_FILE_NAME

    return datasets.read_h5mu_file(training_file)


def load_test_data(split_name: str = DEFAULT_SPLIT_NAME,
                   test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                   seed: int = config.DEFAULT_SEED,
                   subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                   level: str = config.DEFAULT_LEVEL) -> MuData:
    split_dir_path = get_split_dir_path(test_split_size,
                                        seed,
                                        subsample_size,
                                        level)
    test_file = split_dir_path / split_name / TEST_FILE_NAME

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
