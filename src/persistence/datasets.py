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

import logs
from src import config
from persistence import subsampling

log = logs.get_logger()

# Stop warning message
mudata.set_options(pull_on_update=False)

# File prefixes
RNA_FILE_PREFIX = "GSM5008737_RNA"
ADT_FILE_PREFIX = "GSM5008738_ADT"

# Paths
ROOT_PATH = config.ROOT
EXTRACTED_FILE_PATH = config.RAW_DATA_DIR_PATH / "extracted"
PROCESSED_DATA_PATH = config.PROCESSED_DATA_DIR_PATH

# File names
DATASET_FILENAME_TEMPLATE = "multi_modal_dataset__{modifier}.h5mu"
RAW_ARCHIVE_FILE_NAME = "GSE164378_RAW.tar"  # ~1.4 GB
RAW_METADATA_FILE_NAME = "GSE164378_sc.meta.data_3P.csv.gz"

RNA_MATRIX_FILE_NAME = "GSM5008737_RNA_3P-matrix.mtx.gz"
RNA_BARCODES_FILE_NAME = "GSM5008737_RNA_3P-barcodes.tsv.gz"
RNA_FEATURES_FILE_NAME = "GSM5008737_RNA_3P-features.tsv.gz"

ADT_MATRIX_FILE_NAME = "GSM5008738_ADT_3P-matrix.mtx.gz"
ADT_BARCODES_FILE_NAME = "GSM5008738_ADT_3P-barcodes.tsv.gz"
ADT_FEATURES_FILE_NAME = "GSM5008738_ADT_3P-features.tsv.gz"

# File paths
RAW_ARCHIVE_PATH = ROOT_PATH / RAW_ARCHIVE_FILE_NAME
RAW_METADATA_PATH = ROOT_PATH / RAW_METADATA_FILE_NAME

LABLES_TO_DROP = ["Doublet"]


def extract_files_from_main_archive(file_path,
                                    output_dir=EXTRACTED_FILE_PATH):
    log.info("Extracting files from main archive [%s]", file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(file_path) as tar_file:
        for file in tar_file.getmembers():
            if file.name.startswith((RNA_FILE_PREFIX, ADT_FILE_PREFIX)):
                log.debug("Extracting file: %s", file)
                tar_file.extract(file, output_dir)


def get_dataset_file_name(subsample_size=None, seed=None):
    modifier = "full"

    if subsample_size is not None:
        modifier = f"subsample_n_{subsample_size}_seed_{seed}"

    return DATASET_FILENAME_TEMPLATE.format(modifier=modifier)


def read_lines(file_path):
    with gzip.open(file_path, "rt") as file:
        return [line.rstrip("\n") for line in file]


def read_features_file(tsv_file_path):
    tsv_lines = read_lines(tsv_file_path)
    return [line.split("\t")[0] for line in tsv_lines]


def load_rna_features():
    return read_features_file(EXTRACTED_FILE_PATH / RNA_FEATURES_FILE_NAME)


def load_adt_features():
    return read_features_file(EXTRACTED_FILE_PATH / ADT_FEATURES_FILE_NAME)


def load_metadata():
    return pd.read_csv(RAW_METADATA_PATH, index_col=0)


def load_mtx_file(file_path):
    log.info("Loading MatrixMarket matrix from file [%s]", file_path)

    # scipy.io.mmread can handle gzipped files
    matrix = scipy.io.mmread(file_path)

    # Transpose (put cells in the rows) and convert to standard row
    # representation
    log.info("Done loading MatrixMarket matrix from file [%s]", file_path)

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


def save_mudata_dataset_to_disk(dataset,
                                output_file_path):
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write(output_file_path)


def read_h5mu_file(file_path: Path) -> MuData:
    return mudata.read_h5mu(file_path)

def load_or_create_full_dataset(raw_archive_path=RAW_ARCHIVE_PATH,
                                force_recreate=False) -> MuData:
    log.info("Load or create dataset full dataset from raw MatrixMarket files")

    # Check if raw files exist and request download if not
    if not raw_archive_path.exists() or not RAW_METADATA_PATH.exists():
        log.error("Some of the raw persistence files are missing [%s, %s]",
                  RAW_ARCHIVE_FILE_NAME, RAW_METADATA_FILE_NAME)
        log.error("Go to [%s] and download the files.",
                  config.FTP_URL)
        log.error("Place the files as is in the root of the repository.")
        exit(1)

    dataset_file_name = get_dataset_file_name()
    dataset_file = PROCESSED_DATA_PATH / dataset_file_name
    if dataset_file.exists():
        log.info("Dataset exists at %s", dataset_file)
        if force_recreate:
            log.info(f"Recreating dataset [force_recreate={force_recreate}]")
        else:
            log.info(f"Skipping creation [force_recreate={force_recreate}]")
            return read_h5mu_file(dataset_file)
    else:
        log.info("Dataset does not exist.")

    # Extract files from the main archive
    extract_files_from_main_archive(
        file_path=raw_archive_path,
        output_dir=EXTRACTED_FILE_PATH
    )

    # 1. Read raw features
    rna_features = load_rna_features()
    adt_features = load_adt_features()

    # 2. Read raw metadata
    metadata = load_metadata()

    # 3. Read raw persistence
    rna_matrix = load_mtx_file(EXTRACTED_FILE_PATH / RNA_MATRIX_FILE_NAME)
    adt_matrix = load_mtx_file(EXTRACTED_FILE_PATH / ADT_MATRIX_FILE_NAME)

    # 4. Create the dataset in memory
    dataset = create_mudata_dataset(rna_matrix,
                                    adt_matrix,
                                    rna_features,
                                    adt_features,
                                    metadata)

    # 5. Save dataset to disk
    save_mudata_dataset_to_disk(dataset=dataset,
                                output_file_path=dataset_file)

    return dataset


def perliminary_cleanup(dataset, level):
    rna_dataset = dataset["rna"]
    invalid_barcodes = rna_dataset.obs[level].isin(LABLES_TO_DROP)
    return dataset[~invalid_barcodes]


def load_or_create_subsample(subsample_size=config.SUBSAMPLE_SIZE,
                             level=config.PRIMARY_LEVEL,
                             seed=config.SEED,
                             force_recreate=False) -> MuData:
    log.info("Load or create dataset: subsample size=%d, level=%s",
             subsample_size, level)

    dataset_file_name = get_dataset_file_name(subsample_size, seed)
    dataset_file = PROCESSED_DATA_PATH / dataset_file_name
    if dataset_file.exists():
        log.info("Dataset exists at %s", dataset_file)
        if force_recreate:
            log.info(f"Recreating dataset [force_recreate={force_recreate}]")
        else:
            log.info(f"Skipping creation [force_recreate={force_recreate}]")
            return read_h5mu_file(dataset_file)
    else:
        log.info("Dataset does not exist.")

    log.info("Creating dataset with [subsample size=%d] for [level=%s]",
             subsample_size, level)

    # Load or create full dataset.
    # Recreation of the full dataset has to be run explicitly.
    dataset = load_or_create_full_dataset()

    # Cleanup
    dataset = perliminary_cleanup(dataset, level)

    # Subsample
    dataset = subsampling.subsample(dataset=dataset,
                          level=level,
                          subsample_size=subsample_size,
                          seed=seed)

    save_mudata_dataset_to_disk(dataset, dataset_file)

    return dataset
