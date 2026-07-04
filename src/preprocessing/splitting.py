from typing import cast

from mudata import MuData
from sklearn.model_selection import train_test_split

from src import config


def split(dataset: MuData,
          test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
          seed: int = config.DEFAULT_SEED) -> tuple[MuData, MuData]:
    barcodes = dataset.obs_names.to_list()

    barcodes_training_subset, barcodes_test_subset = train_test_split(
        barcodes, test_size=test_split_size / 100, random_state=seed
    )

    training_data = dataset[barcodes_training_subset, :].copy()
    test_data = dataset[barcodes_test_subset, :].copy()

    # We know that the datasets are MuData objects
    return cast(MuData, training_data), cast(MuData, test_data)
