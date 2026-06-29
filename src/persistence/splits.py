from pathlib import Path

from mudata import MuData

from config import PERSISTANCE_DIR
from logs import get_logger
from persistence import datasets

log = get_logger()

SPLITS_ROOT_DIR = PERSISTANCE_DIR / "data_splits"
SPLIT_DIR_NAME_TEMPLATE = "split_{test_split_size}/seed_{seed}"

SUBSAMPLED_DATA_SUBDIR_TEMPLATE = "from_subsampled_dataset/{subsample_size}"
FULL_DATASET_SUBDIR_NAME = "from_full_dataset"

TRAINING_FILE_NAME = "training.h5mu"
TEST_FILE_NAME = "test.h5mu"


def get_split_dir_path(test_split_size: int,
                       seed: int,
                       subsample_size: int = None) -> Path:
    if subsample_size is None:
        subsample_dir = FULL_DATASET_SUBDIR_NAME
    else:
        subsample_dir = SUBSAMPLED_DATA_SUBDIR_TEMPLATE.format(
            subsample_size=subsample_size)

    split_dir_name = SPLIT_DIR_NAME_TEMPLATE.format(
        training_split_size=100 - test_split_size,
        test_split_size=test_split_size,
        seed=seed)

    return SPLITS_ROOT_DIR / subsample_dir / split_dir_name


def save_split(training_data: MuData,
               test_data: MuData,
               test_split_size_pct: int,
               seed: int,
               subsample_size: int = None):
    training_split_size_pct = 100 - test_split_size_pct

    log.info(f"Persist data split to disk (training/test "
             f"{test_split_size_pct}/{training_split_size_pct}, seed: {seed}")

    split_dir_path = get_split_dir_path(test_split_size_pct, seed, subsample_size)
    training_file = split_dir_path / TRAINING_FILE_NAME
    test_file = split_dir_path / TEST_FILE_NAME

    log.info(f"Training data file: {training_file}")
    datasets.save_mudata_dataset_to_disk(training_data, training_file)

    log.info(f"Test data file: {test_file}")
    datasets.save_mudata_dataset_to_disk(test_data, test_file)


def load_training_data(test_split_size_pct: int,
                       seed: int,
                       subsample_size: int = None):
    split_dir_path = get_split_dir_path(test_split_size_pct, seed, subsample_size)
    training_file = split_dir_path / TRAINING_FILE_NAME

    return datasets.read_h5mu_file(training_file)


def load_test_data(test_split_size_pct: int,
                   seed: int,
                   subsample_size: int = None) -> MuData:
    split_dir_path = get_split_dir_path(test_split_size_pct, seed, subsample_size)
    test_file = split_dir_path / TEST_FILE_NAME

    if not test_file.exists():
        log.error(
            f"Split data does not exist (test_split_size: {test_split_size_pct}, seed: {seed})")
        log.error(
            "Check the split size/seed are correct or create the split first.")

    return datasets.read_h5mu_file(test_file)
