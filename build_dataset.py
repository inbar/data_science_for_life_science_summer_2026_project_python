"""Load the raw GSE164378 matrices a MuData object and persist to disk.
This takes a couple of minutes and should be run once locally.

Afterwards the created h5mu dataset can be quickly loaded from disk"""

import time
from src import data, config

t0 = time.time()
print(f"Creating MuData dataset: N={config.SUBSAMPLE_SIZE}, level={config.PRIMARY_LEVEL}")
data = data.create_or_load_dataset(force=True)
print(f"RNA: {data['rna'].shape}  ADT: {data['adt'].shape}")
print("cell types (L2):", data['rna'].obs[config.PRIMARY_LEVEL].value_counts().to_dict())
print(f"DONE in {time.time()-t0:.0f}s")
