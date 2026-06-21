import numpy as np
import scipy.sparse as sp

LAYER_NAME_RAW_COUNTS = "raw_counts"
LAYER_NAME_CENTERED_LOG_RATIO_NORMALIZED = "clr"

def normalize_in_place(dataset):
    """Centered-log-ratio (clr) across proteins within each cell (ADT standard)."""

    # Store unnormalized counts in a layer
    dataset.layers[LAYER_NAME_RAW_COUNTS] = dataset.matrix_of_interest.copy()

    data = dataset.matrix_of_interest.toarray()

    logarithmized_data = np.log1p(data)
    geometric_mean = logarithmized_data.mean(axis=1, keepdims=True)

    clr_normalized_data = sp.csr_matrix(logarithmized_data - geometric_mean).astype(np.float32)

    dataset.matrix_of_interest = clr_normalized_data
    # Store clr normalized data in a layer
    dataset.layers[LAYER_NAME_CENTERED_LOG_RATIO_NORMALIZED] = dataset.matrix_of_interest.copy()
