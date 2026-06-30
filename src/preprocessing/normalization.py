import scanpy as sc

LAYER_NAME_RAW_COUNTS = "raw_counts"
LAYER_NAME_NORMALIZED_COUNTS = "normalized_counts"
LAYER_NAME_LOGARITHMIZED = "logarithmized"

def normalize_in_place(dataset, target_sum=1e4):
    # Store unnormalized counts in a layer
    dataset.layers[LAYER_NAME_RAW_COUNTS] = dataset.X.copy()

    # Normalize the counts for each row to `target_sum`
    sc.pp.normalize_total(dataset, target_sum=target_sum)

    # Store normalized counts in a layer
    dataset.layers[LAYER_NAME_NORMALIZED_COUNTS] = dataset.X.copy()

    # Logarithmize
    sc.pp.log1p(dataset)

    # Store normalized counts in a layer
    dataset.layers[LAYER_NAME_LOGARITHMIZED] = dataset.X.copy()