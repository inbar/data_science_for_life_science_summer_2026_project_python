from pathlib import Path

from src import config

SUBSAMPLED_DATA_SUBDIR_TEMPLATE = "from_subsampled_dataset/{subsample_size}"
FULL_DATASET_SUBDIR_NAME = "from_full_dataset"

SUBDIR_TEMPLATE = "split_{split_size}/seed_{seed}"


def get_subfolder_path(subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                  level: str = config.DEFAULT_LEVEL,
                  test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                  seed: int = config.DEFAULT_SEED) -> Path:
    if subsample_size is config.DEFAULT_SUBSAMPLE_SIZE:
        subsample_dir = FULL_DATASET_SUBDIR_NAME
    else:
        subsample_dir = SUBSAMPLED_DATA_SUBDIR_TEMPLATE.format(
            subsample_size=subsample_size)

    subdir = SUBDIR_TEMPLATE.format(
        split_size=test_split_size,
        seed=seed
    )

    file_path =  Path(level) / subsample_dir / subdir

    return file_path