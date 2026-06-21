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
import logging as log
import tarfile
from array import array

import anndata as ad
import mudata
import numpy as np
import pandas as pd
import scipy.io

from src import config

# Stop warning message
mudata.set_options(pull_on_update=False)

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
rna_features_file_name = "GSM5008737_RNA_3P-features.tsv.gz"

adt_matrix_file_name = "GSM5008738_ADT_3P-matrix.mtx.gz"
adt_barcodes_file_name = "GSM5008738_ADT_3P-barcodes.tsv.gz"
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


def read_rna_features():
    return read_features(extracted_files_path / rna_features_file_name)


def read_adt_features():
    return read_features(extracted_files_path / adt_features_file_name)


def load_and_reduce_metadata(level,
                             rna_barcodes):
    metadata = pd.read_csv(raw_metadata_path, index_col=0)
    # Filter all irrelevant lines from the metadata
    relevant_barcodes = [barcode for barcode in rna_barcodes if
                         barcode in metadata.index]
    metadata = metadata.loc[relevant_barcodes]

    # Drop all rows with unwanted cell lables
    metadata = metadata[~metadata[level].isin(lables_to_drop)]
    return metadata


def select_metadata_indexes_subsample(metadata,
                                      level,
                                      subsample_size,
                                      floor=50,
                                      seed=0):
    """
    Select a subsample using square-root proportional stratified subsampling
    (Square Root Compromise Allocation [Cochran, 1977]).

    Set the sampling fraction of each cell-type proportional to sqrt(n).
    This shrinks the variance between over-represented and minority cell types.
    """
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


def subsample_barcodes(level,
                       metadata,
                       rna_barcodes,
                       subsample_size,
                       seed):
    rna_barcode_to_index_map = {
        barcode: index for index, barcode in enumerate(rna_barcodes)
    }

    subsampled_metadata_indexes = select_metadata_indexes_subsample(metadata,
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


def load_reduced_rna_matrix_by_line(mtx_file_path,
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

    log.info("Loading RNA matrix from file [%s]", mtx_file_path)

    rows = array("i")
    cols = array("i")
    data = array("f")
    with gzip.open(mtx_file_path, "rt") as file:
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


def load_reduced_rna_matrix(mtx_file_path,
                            rna_barcodes,
                            subsampled_barcode_indexes,
                            cell_count,
                            gene_count):
    # This is an array in the size of the original length of the data (rows)
    # Each removed row (cell) contains -1
    # Each kept row (cell) contains a consecutive number to be used as the
    # index in the smaller subset matrix.
    full_barcode_index_to_subset_index_map = np.full(len(rna_barcodes), -1,
                                                     dtype=np.int64)
    full_barcode_index_to_subset_index_map[
        subsampled_barcode_indexes] = np.arange(len(subsampled_barcode_indexes))

    reduced_rna_matrix = load_reduced_rna_matrix_by_line(
        mtx_file_path,
        full_barcode_index_to_subset_index_map,
        cell_count,
        gene_count)

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


def save_mudata_dataset_to_disk(dataset, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write(file_path)


def create_mudata_dataset(metadata,
                          level,
                          rna_matrix,
                          adt_matrix,
                          rna_features,
                          adt_features,
                          subsampled_barcodes,
                          seed):
    observations = metadata.loc[subsampled_barcodes].copy()
    rna_annotated_data = ad.AnnData(X=rna_matrix,
                                    obs=observations.copy(),
                                    var=pd.DataFrame(
                                        index=pd.Index(rna_features,
                                                       name="genes")))

    adt_annotated_data = ad.AnnData(X=adt_matrix,
                                    obs=observations.copy(),
                                    var=pd.DataFrame(
                                        index=pd.Index(adt_features,
                                                       name="proteins")))

    rna_annotated_data.var["gene_name"] = rna_annotated_data.var_names
    rna_annotated_data.var_names_make_unique()

    adt_annotated_data.var["protein_name"] = adt_annotated_data.var_names
    adt_annotated_data.var_names_make_unique()

    dataset = mudata.MuData(
        {"rna": rna_annotated_data, "adt": adt_annotated_data})
    dataset.uns["subsample_level"] = level
    dataset.uns["subsample_seed"] = seed
    dataset.var_names_make_unique()

    return dataset


def create_or_load_dataset(dataset_file=dataset_file_path,
                           subsample_size=config.SUBSAMPLE_SIZE,
                           level=config.PRIMARY_LEVEL,
                           seed=config.SEED,
                           force=False) -> mudata.MuData:
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
        return mudata.read_h5mu(dataset_file)

    # Extract files from the main raw archive
    extract_files_from_main_archive(
        file_path=raw_archive_path,
        output_dir=extracted_files_path
    )

    # Read features
    rna_features = read_rna_features()
    adt_features = read_adt_features()

    # Read barcodes
    rna_barcodes = read_rna_barcodes()
    metadata = load_and_reduce_metadata(level, rna_barcodes)

    # 1. Subsample barcodes
    subsampled_barcodes, subsampled_barcode_indexes = subsample_barcodes(level,
                                                                         metadata,
                                                                         rna_barcodes,
                                                                         subsample_size,
                                                                         seed)
    # 2. Load RNA matrix
    reduced_rna_matrix = load_reduced_rna_matrix(
        extracted_files_path / rna_matrix_file_name,
        rna_barcodes,
        subsampled_barcode_indexes,
        len(subsampled_barcode_indexes),
        len(rna_features)
    )

    # 3. Load ADT matrix
    reduced_adt_matrix = load_reduced_adt_matrix(
        extracted_files_path / adt_matrix_file_name,
        subsampled_barcode_indexes
    )

    # 4. Create the dataset in memory
    dataset = create_mudata_dataset(metadata,
                                    level,
                                    reduced_rna_matrix,
                                    reduced_adt_matrix,
                                    rna_features,
                                    adt_features,
                                    subsampled_barcodes,
                                    seed)

    # 5. Save dataset to disk
    save_mudata_dataset_to_disk(dataset,
                                config.PROCESSED_DATA_DIR_PATH / dataset_file_name)

    return dataset
