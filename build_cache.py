"""One-time heavy build: stream the raw GSE164378 matrices into a cached,
subsampled RNA+ADT MuData. Run once; the notebook then loads the cache."""
import time, sys
sys.path.insert(0, ".")
from src import data_io, config

t0 = time.time()
print(f"Building dataset: N={config.N_SUBSAMPLE}, level={config.PRIMARY_LEVEL}")
mdata = data_io.build_dataset(force=True)
print(mdata)
print(f"RNA: {mdata['rna'].shape}  ADT: {mdata['adt'].shape}")
print("cell types (L2):", mdata['rna'].obs[config.PRIMARY_LEVEL].value_counts().to_dict())
print(f"DONE in {time.time()-t0:.0f}s -> {data_io.CACHE}")
