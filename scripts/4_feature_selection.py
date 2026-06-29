#!/usr/bin/env python

# To run the scripts run:
# source setup_environment.sh
# In the project root

import argparse
import warnings

from anndata import ImplicitModificationWarning
from pandas.errors import PerformanceWarning

from src import config
from src import mappings
from src.logs import get_logger
from src.persistence import splits as split_persistence
from src.persistence.splits import HVG_SPLIT_NAME
from src.preprocessing import rna as rna_preprocessing

warnings.simplefilter("ignore", category=PerformanceWarning)
warnings.simplefilter("ignore", category=ImplicitModificationWarning)

log = get_logger()


def main(args):
    subsample_size = args.subsample_size
    level = args.level
    test_split_size = args.test_split_size
    seed = args.seed

    training_data, test_data = split_persistence.load_split_data(
        test_split_size=test_split_size,
        subsample_size=subsample_size,
        seed=seed,
        level=level
    )

    rna_training_dataset = training_data["rna"]
    adt_training_dataset = training_data["adt"]

    '''
    Feature selection (reduce genes)
    '''
    # Find highly variable genes
    # This must be done only on the training data to prevent data leakage!
    rna_preprocessing.annotate_highly_variable_genes(rna_training_dataset)

    # Extract the gene names
    hv_genes = rna_preprocessing.get_highly_variable_genes(rna_training_dataset)

    # Extract protein names
    protein_names = adt_training_dataset.var["protein_name"]

    marker_genes_for_proteins = mappings.get_marker_genes_for_proteins(
        protein_names)
    all_expressed_genes = set(rna_training_dataset.var["gene_name"])

    # Take the intersection between all expressed genes and the
    # interesting marker genes for the surface proteins
    expressed_marker_genes_for_proteins = (
        all_expressed_genes.intersection(
            marker_genes_for_proteins)

    )

    # From the intersection, take only the high value genes
    genes_of_interest = expressed_marker_genes_for_proteins.union(hv_genes)

    '''
    Filtering the data
    '''
    genes_of_interest_mask = rna_training_dataset.var["gene_name"].isin(
        sorted(genes_of_interest))

    # Subset the RNA modality of the training data
    training_data.mod["rna"] = rna_training_dataset[
        :, genes_of_interest_mask].copy()
    # Update the parent MuData object
    training_data.update()

    # Subset the RNA modality of the test data using the hv genes from
    # the training data
    test_data.mod['rna'] = test_data.mod['rna'][
        :, genes_of_interest_mask].copy()
    # Update the parent MuData object
    test_data.update()

    split_persistence.save_split(training_data=training_data,
                                 test_data=test_data,
                                 split_name=HVG_SPLIT_NAME,
                                 test_split_size=test_split_size,
                                 seed=seed,
                                 subsample_size=subsample_size,
                                 level=level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_size", type=int, default=None)
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--test_split_size", type=str,
                        default=config.DEFAULE_TEST_SPLIT_SIZE)
    parser.add_argument("--seed", type=str, default=config.DEFAULT_SEED)
    parsed_args = parser.parse_args()

    main(parsed_args)
