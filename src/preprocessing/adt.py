import muon as mu
import scanpy as sc

LAYER_NAME_RAW_COUNTS = "raw_counts"
LAYER_NAME_CENTERED_LOG_RATIO = "clr"
LAYER_NAME_LOGARITHMIZED = "logarithmized"

def normalize_in_place(dataset):
    """Centered-log-ratio (clr) across proteins within each cell (ADT standard)."""

    # Store unnormalized counts in a layer
    dataset.layers[LAYER_NAME_RAW_COUNTS] = dataset.X.copy()
    dataset.raw = dataset

    # Logarithmize and store in layer
    dataset.layers[LAYER_NAME_LOGARITHMIZED] = sc.pp.log1p(dataset, copy=True).X


    # Compute CLR
    dataset.X = dataset.X.tocsc().astype('float32')
    mu.prot.pp.clr(dataset)


    # Store clr normalized data in a layer
    dataset.layers[LAYER_NAME_CENTERED_LOG_RATIO] = dataset.X.copy()
