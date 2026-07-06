from typing import cast

from mudata import MuData
from sklearn.model_selection import train_test_split

from src import config


def split(dataset: MuData,
          level: str = config.DEFAULT_LEVEL,
          test_split_size: int = config.DEFAULE_TEST_SPLIT_SIZE,
          seed: int = config.DEFAULT_SEED) -> tuple[MuData, MuData]:
    barcodes = dataset.obs_names.to_list()

    # Stratify by cell type so the train/test split preserves each type's
    # proportion -- an unstratified split can, by chance, drop a rare type (e.g.
    # ASDC, ~76 cells in the full subsample) entirely from train or test.
    #
    # NOTE: read the label from the "rna" modality, not the top-level MuData's
    # own .obs -- datasets.py sets mu.set_options(pull_on_update=False), so the
    # top-level .obs is never populated with per-modality columns like `level`.
    barcodes_training_subset, barcodes_test_subset = train_test_split(
        barcodes, test_size=test_split_size / 100, random_state=seed,
        stratify=dataset["rna"].obs[level]
    )

    training_data = dataset[barcodes_training_subset, :].copy()
    test_data = dataset[barcodes_test_subset, :].copy()

    # We know that the datasets are MuData objects
    return cast(MuData, training_data), cast(MuData, test_data)
