from mudata import MuData
from sklearn.model_selection import train_test_split

from src import config


def split(dataset: MuData,
          test_split_size: float = config.DEFAULE_TEST_SPLIT_SIZE / 100,
          seed: int = config.DEFAULT_SEED):

    barcodes = dataset.obs_names.to_list()

    barcodes_training_subset, barcodes_test_subset = train_test_split(
        barcodes, test_size=test_split_size, random_state=seed
    )

    training_data = dataset[barcodes_training_subset, :].copy()
    test_data = dataset[barcodes_test_subset, :].copy()

    return training_data, test_data