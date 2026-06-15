"""
Load the Hao 2021 (GSE164378) 3' CITE-seq data into a MuData object and
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
from array import array
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import logging as log
from mudata import MuData, read_h5mu
from numpy import dtype, ndarray
from pandas import DataFrame, Index
from scipy.sparse import csr_matrix

import config

# File prefixes
rna_prefix = "GSM5008737_RNA"
adt_prefix = "GSM5008738_ADT"

# Paths
root_path = config.ROOT
extracted_files_path = config.RAW_DATA_DIR_PATH / "extracted"
processed_data_path = config.PROCESSED_DATA_DIR_PATH

# File names
dataset_file_name = "dataset.h5mu"
raw_archive_file_name = "GSE164378_RAW.tar"  # ~1.4 GB
raw_metadata_file_name = "GSE164378_sc.meta.data_3P.csv.gz"

rna_matrix_file_name = "GSM5008737_RNA_3P-matrix.mtx.gz"
rna_barcodes_file_name = "GSM5008737_RNA_3P-barcodes.tsv.gz"
rna_features_file_name = "GSM5008737_RNA_3P-barcodes.tsv.gz"

adt_matrix_file_name = "GSM5008738_ADT_3P-matrix.mtx.gz"
adt_barcodes_file_name = "GSM5008738_ADT_3P-features.tsv.gz"
adt_features_file_name = "GSM5008738_ADT_3P-features.tsv.gz"

# File paths
dataset_file_path = processed_data_path / dataset_file_name
raw_archive_path = root_path / raw_archive_file_name
raw_metadata_path = root_path / raw_metadata_file_name

lables_to_drop = ["Doublet"]


def extract_files_from_main_archive(file_path, output_dir=extracted_files_path):
    log.info("Extracting files from main archive [%s]", file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(file_path) as tar_file:
        for file in tar_file.getmembers():
            if file.name.startswith((rna_prefix, adt_prefix)):
                log.debug("Extracting file: %s", file)
                tar_file.extract(file, output_dir)


def read_lines(file_path):
    with gzip.open(file_path, "rt") as file:
        return [line.rstrip("\n") for line in file]


def read_rna_barcodes():
    return read_lines(extracted_files_path / rna_barcodes_file_name)


def read_features(tsv_file_path):
    tsv_lines = read_lines(tsv_file_path)
    return [line.split("\t")[0] for line in tsv_lines]


def load_and_reduce_metadata(level,
                             rna_barcodes) -> DataFrame:
    metadata = pd.read_csv(raw_metadata_path, index_col=0)
    # Filter all irrelevant lines from the metadata
    relevant_barcodes = [barcode for barcode in rna_barcodes if
                         barcode in metadata.index]
    metadata = metadata.loc[relevant_barcodes]

    # Drop all rows with unwanted cell lables
    metadata = metadata[~metadata[level].isin(lables_to_drop)]
    return metadata


def subsample_barcodes(level,
                       metadata,
                       rna_barcodes,
                       subsample_size,
                       seed):
    rna_barcode_to_index_map = {
        barcode: index for index, barcode in enumerate(rna_barcodes)
    }

    subsampled_metadata_indexes = generate_subsampled_metadata_indexes(metadata,
                                                                       level,
                                                                       subsample_size,
                                                                       seed=seed)

    # Subsample metadata
    subsampled_barcodes = metadata.index[subsampled_metadata_indexes]
    subsampled_barcode_indexes = np.array(
        [rna_barcode_to_index_map[barcode] for barcode in subsampled_barcodes]
    )

    order = np.argsort(subsampled_barcode_indexes)
    subsampled_barcodes = subsampled_barcodes[order]
    subsampled_barcode_indexes = subsampled_barcode_indexes[order]
    return subsampled_barcodes, subsampled_barcode_indexes


def load_reduced_rna_matrix_by_line(file_path,
                                    old_to_new_index_map: np.ndarray,
                                    cell_count: int,
                                    gene_count: int):
    """Load the RNA MatrixMarket file (rows [genes] x cols [cells]),
    keeping selected columns (cells).

    The RNA mtx file is huge (>1G), so we read it line by line to memory
    and only save the cells that are needed.

    `old_to_new_index_map` maps each original 0-based column to its new index
    or -1 if it should be dropped.

    Returns a CSR (Compressed Sparse Row) matrix
      rows [cell_count] x cols [gene_count]
    """

    log.info("Loading RNA matrix from file [%s]", file_path)

    rows = array("i")
    cols = array("i")
    data = array("f")
    with gzip.open(file_path, "rt") as file:
        for line in file:
            if line.startswith("%"):
                continue
            # The break skips over the dimensions line
            break
        for line in file:
            row, column, value = line.split()
            new_row_index = old_to_new_index_map[int(column) - 1]
            if new_row_index != -1:
                cols.append(int(row) - 1)  # gene -> column of output
                rows.append(new_row_index)  # cell -> row of output
                data.append(float(value))

    log.info("Done loading RNA matrix")

    rows = np.frombuffer(rows, dtype=np.int32)
    cols = np.frombuffer(cols, dtype=np.int32)

    data = np.frombuffer(data, dtype=np.float32)
    coords = (rows, cols)
    sparse_matrix = scipy.sparse.coo_matrix(
        (data, coords), shape=(cell_count, gene_count)
    )

    return sparse_matrix.tocsr()


def load_reduced_rna_matrix(file_path,
                            rna_barcodes,
                            subsampled_barcode_indexes):
    # This is an array in the size of the original length of the data (rows)
    # Each removed row (cell) contains -1
    # Each kept row (cell) contains a consecutive number to be used as the
    # index in the smaller subset matrix.
    full_barcode_index_to_subset_index_map = np.full(len(rna_barcodes), -1,
                                                     dtype=np.int64)
    full_barcode_index_to_subset_index_map[
        subsampled_barcode_indexes] = np.arange(len(subsampled_barcode_indexes))

    reduced_rna_matrix = load_reduced_rna_matrix_by_line(
        file_path,
        full_barcode_index_to_subset_index_map,
        len(subsampled_barcode_indexes),
        len(rna_barcodes))

    return reduced_rna_matrix


def load_reduced_adt_matrix(file_path, subsampled_barcodes_indexes):
    """Load the ADT MatrixMarket file (rows [genes] x cols [cells]),
       keeping selected columns (cells).

       The ADT mtx file is not as large as the RNA one, so we can load it as is
       to memory and drop the unneeded cells.

       `subsampled_barcodes_indexes` are the indexes of the cells (columns) to
       keep.

       Returns a CSR (Compressed Sparse Row) matrix
         rows [cell_count] x cols [gene_count]
       """

    log.info("Loading ADT matrix from file [%s]", file_path)
    # scipy.io.mmread can handle gzipped files
    matrix = scipy.io.mmread(file_path)

    # The raw mtx has the cells in the columns.
    # To reduce the matrix to the cell subsample, we need to:

    # 1. Convert to compressed sparse columns representation
    # for better column splicing
    matrix = matrix.tocsc()

    # 2. Reduce columns to cell subsample
    matrix = matrix[:, subsampled_barcodes_indexes]

    # Transpose (put cells in the rows) and convert to standard row
    # representation

    log.info("Done loading ADT matrix")

    return matrix.T.tocsr()


def generate_subsampled_metadata_indexes(metadata,
                                         level,
                                         subsample_size,
                                         floor: int = 50,
                                         seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    level_values = metadata[level]
    value_counts = level_values.value_counts()
    sqrt_value_counts = np.sqrt(value_counts.astype(float))

    # Compute the proportional share of each value
    proportional_counts = subsample_size * (
        sqrt_value_counts / sqrt_value_counts.sum()
    )

    # Floor and cap the proportion
    proportional_counts = np.maximum(proportional_counts, floor)
    proportional_counts = np.minimum(proportional_counts,
                                     value_counts.values).round().astype(int)

    subsample_per_label = []
    label_positions = {lab: np.where(level_values.values == lab)[0] for lab in
                       value_counts.index}

    # Randomly select proportional_count number of samples from each label
    for label, proportional_count in zip(value_counts.index,
                                         proportional_counts):
        all_indexes_for_label = label_positions[label]
        size = min(proportional_count, len(all_indexes_for_label))
        subsample_per_label.append(
            rng.choice(all_indexes_for_label,
                       size=size, replace=False)
        )

    # Return a sorted list of all indexes to keep (the subsample)
    return np.sort(np.concatenate(subsample_per_label))


def save_mudata_dataset_to_disk(mudata: MuData, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mudata.write(file_path)


def create_mudata_dataset(metadata,
                          level,
                          rna_matrix,
                          adt_matrix,
                          subsampled_barcodes,
                          seed) -> MuData:
    rna_features = read_features(extracted_files_path / rna_features_file_name)
    observations = metadata.loc[subsampled_barcodes].copy()
    rna_annotated_data = ad.AnnData(X=rna_matrix,
                                    obs=observations.copy(),
                                    var=pd.DataFrame(
                                        index=pd.Index(rna_features,
                                                       name="gene")))

    adt_features = read_features(extracted_files_path / adt_features_file_name)
    adt_annotated_data = ad.AnnData(X=adt_matrix,
                                    obs=observations.copy(),
                                    var=pd.DataFrame(
                                        index=pd.Index(adt_features,
                                                       name="protein")))

    rna_annotated_data.var_names_make_unique()
    adt_annotated_data.var_names_make_unique()

    mdata = MuData({"rna": rna_annotated_data, "adt": adt_annotated_data})
    mdata.uns["subsample_level"] = level
    mdata.uns["subsample_seed"] = seed

    return mdata


def create_or_load_dataset(dataset_file=dataset_file_path,
                           subsample_size=config.SUBSAMPLE_SIZE,
                           level=config.PRIMARY_LEVEL,
                           seed=config.SEED,
                           force=False) -> MuData:
    # Check if raw files exist and request download if not
    if not raw_archive_path.exists() or not raw_metadata_path.exists():
        log.error("Some of the raw data files are missing [%s, %s]",
                  raw_archive_file_name, raw_metadata_file_name)
        log.error("Go to [%s] and download the files.",
                  config.FTP_URL)
        log.error("Place the files as is in the root of the repository.")
        exit(1)

    # Try to load the processed dataset (h5mu file) if it exists
    if dataset_file.exists() and not force:
        log.info("Dataset exists at %s", dataset_file)
        log.info("Loading from file [%s]", dataset_file)
        return read_h5mu(dataset_file)

    # Extract files from the main raw archive
    extract_files_from_main_archive(
        file_path=raw_archive_path,
        output_dir=extracted_files_path
    )

    rna_barcodes = read_rna_barcodes()
    metadata = load_and_reduce_metadata(level, rna_barcodes)

    # 1. Subsample barcodes
    subsampled_barcodes, subsampled_barcode_indexes = subsample_barcodes(level,
                                                                         metadata,
                                                                         rna_barcodes,
                                                                         seed,
                                                                         subsample_size)

    # 2. Load RNA matrix    
    reduced_rna_matrix = load_reduced_rna_matrix(
        extracted_files_path / rna_matrix_file_name,
        rna_barcodes,
        subsampled_barcode_indexes)

    # 3. Load ADT matrix
    reduced_adt_matrix = load_reduced_adt_matrix(
        extracted_files_path / adt_matrix_file_name,
        subsampled_barcode_indexes
    )

    # 4. Create the dataset in memory
    mudata = create_mudata_dataset(metadata,
                                   level,
                                   reduced_rna_matrix,
                                   reduced_adt_matrix,
                                   subsampled_barcodes,
                                   seed)

    # 5. Save dataset to disk
    save_mudata_dataset_to_disk(mudata,
                                config.PROCESSED_DATA_DIR_PATH / dataset_file_name)

    return mudata


create_or_load_dataset()
