"""Load the Hao 2021 (GSE164378) 3' CITE-seq data into a cached MuData object.

The raw RNA matrix is 33,538 genes x 161,764 cells (354M non-zeros), too large to
hold comfortably in RAM alongside downstream work. We therefore (1) choose the
stratified cell subsample first, then (2) *stream* the MatrixMarket file keeping
only those cells' columns, so peak memory stays small. The ADT matrix (228
proteins) is small and read in full. The result is cached to ``data/processed``.

Subsampling decision
---------------------
We use **sqrt-proportional** stratified sampling on the primary annotation level:
each cell type gets ~ sqrt(size) share (floored, capped at its true size), scaled
to the target N. Pure proportional sampling would leave rare-but-genuine PBMC
types (ILC, cDC1, HSPC) with too few cells to score; sqrt allocation keeps them in
the benchmark while common types still dominate. Doublets are dropped.
"""
from __future__ import annotations

import gzip
import os
import tarfile
import urllib.request
from array import array

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from mudata import MuData

from . import config

RAW = config.DATA_RAW
EXT = RAW / "extracted"
CACHE = config.DATA_PROC / "pbmc3p_citeseq.h5mu"

RNA_PREFIX = "GSM5008737_RNA_3P"
ADT_PREFIX = "GSM5008738_ADT_3P"
META_FILE = "GSE164378_sc.meta.data_3P.csv.gz"
DROP_LABELS = ("Doublet",)

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/"
RAW_TAR = "GSE164378_RAW.tar"          # ~1.4 GB (RNA+ADT+HTO count matrices)


# --------------------------------------------------------------------------- #
# Raw file helpers
# --------------------------------------------------------------------------- #
def _download(fn: str):
    """Stream a GEO supplementary file to data/raw (resumable, skips if done)."""
    dest = RAW / fn
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(GEO_BASE + fn)
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            return  # already complete
        raise
    mode = "ab" if have and resp.status == 206 else "wb"
    with open(dest, mode) as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)


def _ensure_raw():
    for fn in (RAW_TAR, META_FILE):
        if not (RAW / fn).exists():
            print(f"downloading {fn} ...", flush=True)
            _download(fn)


def _extract_3p():
    _ensure_raw()
    EXT.mkdir(parents=True, exist_ok=True)
    needed = [f"{p}-{s}" for p in (RNA_PREFIX, ADT_PREFIX)
              for s in ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")]
    if all((EXT / n).exists() for n in needed):
        return
    with tarfile.open(RAW / "GSE164378_RAW.tar") as t:
        for m in t.getmembers():
            if m.name.startswith((RNA_PREFIX, ADT_PREFIX)):
                t.extract(m, EXT)


def _read_lines(path) -> list[str]:
    with gzip.open(path, "rt") as f:
        return [ln.rstrip("\n") for ln in f]


def _read_features(prefix) -> list[str]:
    # features.tsv: col0 = symbol (RNA) / antibody name (ADT)
    return [ln.split("\t")[0] for ln in _read_lines(EXT / f"{prefix}-features.tsv.gz")]


def _read_barcodes(prefix) -> list[str]:
    return _read_lines(EXT / f"{prefix}-barcodes.tsv.gz")


def _stream_mtx_columns(path, keep_newcol: np.ndarray, n_new: int, n_rows: int):
    """Stream a genes x cells MatrixMarket file, keeping selected cell columns.

    ``keep_newcol`` maps each original 0-based column to its new index (or -1).
    Returns a CSR matrix of shape (n_new_cells, n_rows_genes).
    """
    rows = array("i"); cols = array("i"); data = array("f")
    with gzip.open(path, "rt") as f:
        for ln in f:
            if ln[0] == "%":
                continue
            break  # the dims line (already known) is consumed here
        for ln in f:
            r, c, v = ln.split()
            nc = keep_newcol[int(c) - 1]
            if nc >= 0:
                cols.append(int(r) - 1)   # gene -> column of output
                rows.append(nc)           # cell -> row of output
                data.append(float(v))
    m = sp.coo_matrix(
        (np.frombuffer(data, dtype=np.float32),
         (np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32))),
        shape=(n_new, n_rows),
    )
    return m.tocsr()


# --------------------------------------------------------------------------- #
# Subsampling
# --------------------------------------------------------------------------- #
def sqrt_proportional_indices(labels: pd.Series, n_target: int, floor: int = 50,
                              seed: int = 0) -> np.ndarray:
    """Indices (into ``labels``) for a sqrt-proportional stratified subsample."""
    rng = np.random.default_rng(seed)
    sizes = labels.value_counts()
    weights = np.sqrt(sizes.astype(float))
    alloc = (weights / weights.sum() * n_target)
    alloc = np.maximum(alloc, floor)
    alloc = np.minimum(alloc, sizes.values).round().astype(int)
    chosen = []
    pos = {lab: np.where(labels.values == lab)[0] for lab in sizes.index}
    for lab, k in zip(sizes.index, alloc):
        idx = pos[lab]
        chosen.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return np.sort(np.concatenate(chosen))


# --------------------------------------------------------------------------- #
# Build / load
# --------------------------------------------------------------------------- #
def build_dataset(n_subsample: int = config.N_SUBSAMPLE,
                  level: str = config.PRIMARY_LEVEL,
                  seed: int = config.SEED,
                  force: bool = False) -> MuData:
    """Build (and cache) the subsampled RNA+ADT MuData."""
    if CACHE.exists() and not force:
        return load_dataset()

    _extract_3p()
    barcodes = _read_barcodes(RNA_PREFIX)
    assert _read_barcodes(ADT_PREFIX) == barcodes, "RNA/ADT barcode order differs"

    meta = pd.read_csv(RAW / META_FILE, index_col=0)
    meta = meta.loc[[b for b in barcodes if b in meta.index]]   # align to file order
    # boolean over file-order barcodes that are in metadata and not dropped
    bc_pos = {b: i for i, b in enumerate(barcodes)}
    keep_label = meta[~meta[level].isin(DROP_LABELS)]
    sub_local = sqrt_proportional_indices(keep_label[level], n_subsample, seed=seed)
    sub_barcodes = keep_label.index[sub_local]
    sub_global = np.array([bc_pos[b] for b in sub_barcodes])

    order = np.argsort(sub_global)
    sub_global = sub_global[order]
    sub_barcodes = sub_barcodes[order]

    keep_newcol = np.full(len(barcodes), -1, dtype=np.int64)
    keep_newcol[sub_global] = np.arange(len(sub_global))

    rna_genes = _read_features(RNA_PREFIX)
    adt_prot = _read_features(ADT_PREFIX)

    X_rna = _stream_mtx_columns(EXT / f"{RNA_PREFIX}-matrix.mtx.gz",
                                keep_newcol, len(sub_global), len(rna_genes))

    adt_full = scipy.io.mmread(gzip.open(EXT / f"{ADT_PREFIX}-matrix.mtx.gz", "rb"))
    X_adt = adt_full.tocsc()[:, sub_global].T.tocsr()  # cells x proteins

    obs = meta.loc[sub_barcodes].copy()
    rna = ad.AnnData(X=X_rna, obs=obs,
                     var=pd.DataFrame(index=pd.Index(rna_genes, name="gene")))
    rna.var_names_make_unique()
    adt = ad.AnnData(X=np.asarray(X_adt.todense(), dtype=np.float32), obs=obs.copy(),
                     var=pd.DataFrame(index=pd.Index(adt_prot, name="protein")))
    adt.var_names_make_unique()

    mdata = MuData({"rna": rna, "adt": adt})
    mdata.uns["subsample_level"] = level
    mdata.uns["subsample_seed"] = seed
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    mdata.write(CACHE)
    return mdata


def load_dataset() -> MuData:
    from mudata import read_h5mu
    return read_h5mu(CACHE)
