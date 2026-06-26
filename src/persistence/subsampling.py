import numpy as np
from mudata import MuData

import config
from src import logs

log = logs.get_logger()


def subsample_barcodes(rna_dataset,
                       level,
                       subsample_size,
                       floor=50,
                       seed=config.SEED):
    """
    Select a subsample using square-root proportional stratified subsampling
    (Square Root Compromise Allocation [Cochran, 1977]).

    Set the sampling fraction of each cell-type proportional to sqrt(n).
    This shrinks the variance between over-represented and minority cell types.
    """
    log.debug("Subsampling barcodes: level=%s, subsample_size=%d, seed=%d",
              level, subsample_size, seed)
    rng = np.random.default_rng(seed)
    level_values = rna_dataset.obs[level]
    level_value_counts = level_values.value_counts()
    sqrt_value_counts = np.sqrt(level_value_counts.astype(float))

    # Compute the proportional share of each value
    proportional_counts = subsample_size * (
        sqrt_value_counts / sqrt_value_counts.sum()
    )

    # Floor and cap the proportion
    proportional_counts = np.maximum(proportional_counts, floor)
    proportional_counts = np.minimum(proportional_counts,
                                     level_value_counts.values).round().astype(
        int)

    label_positions = {
        level_value: np.where(level_values.values == level_value)[0] for
        level_value in level_value_counts.index}

    # Randomly select proportional_count number of samples from each label
    subsample_per_level = []
    for label, proportional_count in zip(level_value_counts.index,
                                         proportional_counts):
        all_indexes_for_label = label_positions[label]
        size = min(proportional_count, len(all_indexes_for_label))
        subsample_per_level.append(
            rng.choice(all_indexes_for_label,
                       size=size, replace=False)
        )

    indexes_to_keep = np.sort(np.concatenate(subsample_per_level))
    barcodes_to_keep = rna_dataset.obs.iloc[indexes_to_keep].index.tolist()
    return barcodes_to_keep


def subsample(dataset: MuData,
              level,
              subsample_size,
              seed=config.SEED):
    barcodes_to_keep = subsample_barcodes(rna_dataset=dataset["rna"],
                                          level=level,
                                          subsample_size=subsample_size,
                                          seed=seed)

    return dataset[barcodes_to_keep].copy()
