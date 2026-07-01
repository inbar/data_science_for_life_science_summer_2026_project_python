from pathlib import Path

from src import config

SUBSAMPLE_SUBDIR_PREFIX_TEMPLATE = "subsample_{subsample_size}"
FULL_DATASET_SUBDIR_PREFIX_NAME = "full_dataset"

SUBDIR_TEMPLATE = "split_{split_size}_seed_{seed}"


def get_subfolder_path(subsample_size: int = config.DEFAULT_SUBSAMPLE_SIZE,
                  level: str = config.DEFAULT_LEVEL,
                  test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
                  seed: int = config.DEFAULT_SEED) -> Path:
    if subsample_size is config.DEFAULT_SUBSAMPLE_SIZE:
        subdir_prefix = FULL_DATASET_SUBDIR_PREFIX_NAME
    else:
        subdir_prefix = SUBSAMPLE_SUBDIR_PREFIX_TEMPLATE.format(
            subsample_size=subsample_size)

    subdir = SUBDIR_TEMPLATE.format(
        split_size=test_split_size,
        seed=seed
    )

    file_path =  Path(level) / f"{subdir_prefix}_{subdir}"

    return file_path