"""
Load the Hao 2021 (GSE164378) 3' CITE-seq persistence into a MuData object and
save it locally.

The raw RNA matrix is 33,538 genes x 161,764 cells (354M non-zeros), too large to
hold comfortably in RAM alongside downstream work. We therefore
(1) choose the stratified cell subsample first
(2) stream the MatrixMarket file keeping only those cells' columns, so peak memory stays small.

The ADT matrix (228 proteins) is small and read in full.


Subsampling decision
---------------------
We use "sqrt-proportional" stratified sampling on the primary annotation level:
each cell type gets ~ sqrt(size) share (floored, capped at its true size), scaled
to the target N. Pure proportional sampling would leave rare-but-genuine PBMC
types (ILC, cDC1, HSPC) with too few cells to score; sqrt allocation keeps them in
the benchmark while common types still dominate. Doublets are dropped.

"""
from __future__ import annotations

import gzip
import tarfile
from pathlib import Path

import anndata as ad
import mudata
import pandas as pd
import scipy.io
from mudata import MuData

from src.persistence import subsampling
from src import config

import logging

log = logging.getLogger(__file__)

# Avoid warning message
mudata.set_options(pull_on_update=False)

# File prefixes
RNA_FILE_PREFIX = "GSM5008737_RNA"
ADT_FILE_PREFIX = "GSM5008738_ADT"

# Raw files
# File names
RAW_ARCHIVE_FILE_NAME = "GSE164378_RAW.tar"  # ~1.4 GB
RAW_METADATA_FILE_NAME = "GSE164378_sc.meta.data_3P.csv.gz"

RNA_MATRIX_FILE_NAME = "GSM5008737_RNA_3P-matrix.mtx.gz"
RNA_BARCODES_FILE_NAME = "GSM5008737_RNA_3P-barcodes.tsv.gz"
RNA_FEATURES_FILE_NAME = "GSM5008737_RNA_3P-features.tsv.gz"

ADT_MATRIX_FILE_NAME = "GSM5008738_ADT_3P-matrix.mtx.gz"
ADT_BARCODES_FILE_NAME = "GSM5008738_ADT_3P-barcodes.tsv.gz"
ADT_FEATURES_FILE_NAME = "GSM5008738_ADT_3P-features.tsv.gz"

# File paths
RAW_ARCHIVE_PATH = config.PROJECT_REPO_ROOT / RAW_ARCHIVE_FILE_NAME
RAW_METADATA_PATH = config.PROJECT_REPO_ROOT / RAW_METADATA_FILE_NAME

# Processed files
# Dir Paths
FULL_DATASETS_ROOT_DIR = config.PROCESSED_DATA / "full_datasets"
SUBSAMPLE_DATASETS_ROOT_DIR = config.PROCESSED_DATA / "subsampled_datasets"
SUBSAMPLE_DATASET_SUBDIR_TEMPLATE = "subsample_{subsample_size}/seed_{seed}"

# File names
DATASET_FILENAME = "multi_modal_dataset.h5mu"


def extract_files_from_main_archive(file_path,
                                    output_dir=config.RAW_DATA_DIR):
    log.info(f"Extracting files from main archive [{file_path}]")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(file_path) as tar_file:
        for file in tar_file.getmembers():
            if file.name.startswith((RNA_FILE_PREFIX, ADT_FILE_PREFIX)):
                log.debug(f"Extracting file: {file}")
                tar_file.extract(file, output_dir)


def get_dataset_file_path(subsample_size=config.DEFAULT_SUBSAMPLE_SIZE,
                          seed=config.DEFAULT_SEED,
                          level=config.DEFAULT_LEVEL) -> Path:
    if subsample_size is config.DEFAULT_SUBSAMPLE_SIZE:
        return FULL_DATASETS_ROOT_DIR / level / DATASET_FILENAME

    subsample_dataset_dir = SUBSAMPLE_DATASET_SUBDIR_TEMPLATE.format(
        subsample_size=subsample_size, seed=seed)

    return SUBSAMPLE_DATASETS_ROOT_DIR / level / subsample_dataset_dir / DATASET_FILENAME


def read_lines(file_path):
    with gzip.open(file_path, "rt") as file:
        return [line.rstrip("\n") for line in file]


def read_features_file(tsv_file_path):
    tsv_lines = read_lines(tsv_file_path)
    return [line.split("\t")[0] for line in tsv_lines]


def load_rna_features():
    return read_features_file(config.RAW_DATA_DIR / RNA_FEATURES_FILE_NAME)


def load_adt_features():
    return read_features_file(config.RAW_DATA_DIR / ADT_FEATURES_FILE_NAME)


def load_metadata():
    return pd.read_csv(RAW_METADATA_PATH, index_col=0)


def load_mtx_file(file_path):
    log.info(f"Loading MatrixMarket matrix from file [{file_path}]")

    # scipy.io.mmread can handle gzipped files
    matrix = scipy.io.mmread(file_path)

    # Transpose (put cells in the rows) and convert to standard row
    # representation
    log.info(f"Done loading MatrixMarket matrix from file [{file_path}]")

    return matrix.T.tocsr()


def create_mudata_dataset(rna_matrix,
                          adt_matrix,
                          rna_features,
                          adt_features,
                          metadata):
    rna_annotated_data = ad.AnnData(X=rna_matrix,
                                    obs=metadata.copy(),
                                    var=pd.DataFrame(
                                        index=pd.Index(rna_features,
                                                       name="genes")))

    adt_annotated_data = ad.AnnData(X=adt_matrix,
                                    obs=metadata.copy(),
                                    var=pd.DataFrame(
                                        index=pd.Index(adt_features,
                                                       name="proteins")))

    rna_annotated_data.var["gene_name"] = rna_annotated_data.var_names
    rna_annotated_data.var_names_make_unique()

    adt_annotated_data.var["protein_name"] = adt_annotated_data.var_names
    adt_annotated_data.var_names_make_unique()

    dataset = mudata.MuData(
        {"rna": rna_annotated_data, "adt": adt_annotated_data})
    dataset.var_names_make_unique()

    return dataset


def save_mudata_dataset_to_disk(dataset: MuData,
                                output_file_path: Path):
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write(output_file_path)


def save_full_dataset(dataset: MuData, level: str = config.DEFAULT_LEVEL):
    dataset_file = get_dataset_file_path(level=level)
    save_mudata_dataset_to_disk(dataset, dataset_file)


def load_full_dataset(level: str = config.DEFAULT_LEVEL):
    dataset_file = get_dataset_file_path(level=level)

    return read_h5mu_file(dataset_file)


def save_subsampled_dataset(dataset: MuData,
                            subsample_size=-1,
                            seed=config.DEFAULT_SEED,
                            level: str = config.DEFAULT_LEVEL):
    dataset_file = get_dataset_file_path(subsample_size=subsample_size,
                                         seed=seed,
                                         level=level)
    save_mudata_dataset_to_disk(dataset, dataset_file)


def load_subsampled_dataset(subsample_size=-1,
                            seed=config.DEFAULT_SEED,
                            level: str = config.DEFAULT_LEVEL):
    dataset_file = get_dataset_file_path(subsample_size=subsample_size,
                                         seed=seed,
                                         level=level)
    return read_h5mu_file(dataset_file)


def dataset_exist(subsample_size=config.DEFAULT_SUBSAMPLE_SIZE,
                  seed: int = config.DEFAULT_SEED,
                  level: str = config.DEFAULT_LEVEL):
    dataset_file = get_dataset_file_path(subsample_size=subsample_size,
                                         seed=seed,
                                         level=level)
    exists = dataset_file.exists()

    if exists:
        log.info(f"Dataset exists: {dataset_file}")

    return dataset_file.exists()


def read_h5mu_file(file_path: Path) -> MuData:
    return mudata.read_h5mu(file_path)


def load_or_create_full_dataset(raw_archive_path=RAW_ARCHIVE_PATH,
                                force_recreate=False) -> MuData:
    log.info("Load or create dataset full dataset from raw MatrixMarket files")

    # Check if raw files exist and request download if not
    if not raw_archive_path.exists() or not RAW_METADATA_PATH.exists():
        log.error(
            f"Some of the raw persistence files are missing [{RAW_ARCHIVE_FILE_NAME}, {RAW_METADATA_FILE_NAME}]")
        log.error(f"Go to {config.FTP_URL} and download the files.")
        log.error("Place the files as is in the root of the repository.")
        raise FileNotFoundError(raw_archive_path)

    if dataset_exist():
        if force_recreate:
            log.info(f"Recreating dataset [force_recreate={force_recreate}]")
        else:
            log.info(f"Skipping creation [force_recreate={force_recreate}]")
            return load_full_dataset()

    # Extract files from the main archive
    extract_files_from_main_archive(
        file_path=raw_archive_path,
        output_dir=config.RAW_DATA_DIR
    )

    # 1. Read raw features
    rna_features = load_rna_features()
    adt_features = load_adt_features()

    # 2. Read raw metadata
    metadata = load_metadata()

    # 3. Read raw persistence
    rna_matrix = load_mtx_file(config.RAW_DATA_DIR / RNA_MATRIX_FILE_NAME)
    adt_matrix = load_mtx_file(config.RAW_DATA_DIR / ADT_MATRIX_FILE_NAME)

    # 4. Create the dataset in memory
    dataset = create_mudata_dataset(rna_matrix,
                                    adt_matrix,
                                    rna_features,
                                    adt_features,
                                    metadata)

    return dataset


def load_or_create_subsample(subsample_size=config.DEFAULT_SUBSAMPLE_SIZE,
                             level=config.DEFAULT_LEVEL,
                             seed=config.DEFAULT_SEED,
                             force_recreate=False) -> MuData:
    log.info(
        f"Load or create dataset: subsample size={subsample_size}, level={level}")

    if dataset_exist(subsample_size, seed):
        if force_recreate:
            log.info(f"Recreating dataset [force_recreate={force_recreate}]")
        else:
            log.info(f"Skipping creation [force_recreate={force_recreate}]")
            return load_subsampled_dataset(subsample_size, seed)

    log.info(
        f"Creating subsample dataset: (subsample size={subsample_size}, level={level})")

    # Load or create full dataset.
    # For recreating the full dataset, it has to be run separately.
    dataset = load_or_create_full_dataset()

    # Subsample
    dataset = subsampling.subsample(dataset=dataset,
                                    level=level,
                                    subsample_size=subsample_size,
                                    seed=seed)

    return dataset
