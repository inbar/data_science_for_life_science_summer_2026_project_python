#!/usr/bin/env python
import argparse




# To run the scripts run:
# source setup_environment.sh
# In the project root

from src.persistence import datasets as dataset_persistence
from src import config

from src.logs import get_logger

import warnings
from pandas.errors import PerformanceWarning
from anndata import ImplicitModificationWarning
from src.preprocessing import normalization
from src.preprocessing import rna as rna_preprocessing


warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

log = get_logger()


def main(args):
    level = args.level

    dataset = dataset_persistence.load_or_create_full_dataset()
    rna_dataset, adt_dataset = dataset["rna"], dataset["adt"]

    # Normalize
    rna_preprocessing.calculate_qc_metrics_in_place(rna_dataset)
    normalization.normalize_in_place(rna_dataset)
    normalization.normalize_in_place(adt_dataset)

    # Basic cell filtering
    rna_dataset_filtered = rna_preprocessing.apply_basic_filtering(rna_dataset, level)

    dataset.mod["rna"] = rna_dataset_filtered
    dataset.mod["adt"] = adt_dataset[dataset["rna"].obs_names, :].copy()

    dataset.update()

    dataset_persistence.save_full_dataset(dataset, level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parsed_args = parser.parse_args()

    main(parsed_args)
