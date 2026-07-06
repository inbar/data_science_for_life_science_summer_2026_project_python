#!/usr/bin/env python
"""Dev/validation-only substitute for ``1_load_full_dataset.py``.

This checkout has no local copy of the raw GEO archive (``GSE164378_RAW.tar`` /
``GSE164378_sc.meta.data_3P.csv.gz``), so the normal ``dataset_persistence.
load_or_create_full_dataset()`` path cannot run here. This script instead reads
an already-processed MuData (produced by a sibling pipeline) and applies the
IDENTICAL preprocessing as ``1_load_full_dataset.py`` -- it imports that script's
``preprocess_and_filter`` function directly rather than reimplementing it, so the
two can never drift apart -- then writes the result to the SAME persistence
cache path ``1_load_full_dataset.py`` would have written to.

After running this once (per --level), every downstream script in the normal
numbered pipeline (2_create_subsample_datasets.py, 3_exploratory_analysis.py,
4_split_dataset.py, 5_feature_selection.py, 6_ground_truth.py, 7_scoring.py, and
scripts/deep_learning/*) runs completely unmodified, exactly as it would on a
machine with the real raw data.

# To run the scripts run:
# source setup_environment.sh
# In the project root
"""
import argparse
import importlib.util
import os
import warnings
from pathlib import Path

import mudata as md
from anndata import ImplicitModificationWarning
from pandas.errors import PerformanceWarning

from src import config
from src import logs
from src.persistence import datasets as dataset_persistence

warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

import logging

logs.setup_logging(__file__)
log = logging.getLogger(__file__)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_H5MU_PATH = os.environ.get("VALIDATION_H5MU", "")


def _import_load_full_dataset():
    """Import the numbered ``1_load_full_dataset.py`` script as a module (its
    filename is not a valid identifier, so a normal ``import`` cannot reach it)."""
    spec = importlib.util.spec_from_file_location(
        "load_full_dataset", SCRIPTS_DIR / "1_load_full_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    level = args.level
    h5mu_path = args.h5mu_path
    force_recreate = args.force_recreate

    log.info(f"Seed full dataset from external h5mu (dev/validation substitute "
             f"for 1_load_full_dataset.py)")
    log.info("=" * 70)
    for k, v in vars(parsed_args).items():
        log.info(f"   {k}: {v}")
    log.info("")

    if dataset_persistence.dataset_exist(level=level) and not force_recreate:
        log.info("Full dataset already cached; skipping (--force_recreate to redo).")
        return

    load_full_dataset = _import_load_full_dataset()

    log.info(f"Loading {h5mu_path}")
    mdata = md.read_h5mu(h5mu_path)
    rna_dataset, adt_dataset = mdata["rna"].copy(), mdata["adt"].copy()
    rna_dataset.var["gene_name"] = rna_dataset.var_names.astype(str)
    adt_dataset.var["protein_name"] = adt_dataset.var_names.astype(str)
    log.info(f"{rna_dataset.n_obs} cells | RNA {rna_dataset.n_vars} genes | "
             f"ADT {adt_dataset.n_vars} proteins")

    filtered_dataset = load_full_dataset.preprocess_and_filter(
        rna_dataset, adt_dataset, level)

    dataset_persistence.save_full_dataset(filtered_dataset, level)
    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--h5mu_path", type=str, default=DEFAULT_H5MU_PATH,
                        required=not DEFAULT_H5MU_PATH,
                        help="Path to a processed MuData (.h5mu) with 'rna' and "
                             "'adt' modalities to seed the full-dataset cache "
                             "from. Defaults to the VALIDATION_H5MU env var.")
    parser.add_argument("--force_recreate", type=bool, default=False)
    parsed_args = parser.parse_args()

    main(parsed_args)
