"""
Persistence for the GO-enrichment / functional-annotation artifacts produced
by scripts/8_annotation_analysis.py: the three tidy enrichment tables (full
ranking, method-unique gene sets, axis-unique gene sets) plus the raw gene
sets they were computed on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config
from src.persistence import path_tools

import logging

log = logging.getLogger(__file__)

ENRICHMENT_FULL_FILENAME = "enrichment_full_ranking.csv"
ENRICHMENT_UNIQUE_FILENAME = "enrichment_method_unique.csv"
ENRICHMENT_AXIS_FILENAME = "enrichment_axis_unique.csv"
GENE_SETS_FILENAME = "gene_sets.json"
GPROFILER_CACHE_SUBDIR = "gprofiler_cache"


def get_annotation_dir(subsample_size=config.DEFAULT_SUBSAMPLE_SIZE,
                       level: str = config.DEFAULT_LEVEL,
                       test_split_size: int = None,
                       seed: int = config.DEFAULT_SEED,
                       root_dir: Path = config.LOCAL_DATA_ROOT) -> Path:

    subdir = path_tools.get_subfolder_path(
        subsample_size=subsample_size, level=level,
        test_split_size=test_split_size, seed=seed)
    return Path(root_dir) / "annotation" / subdir


def get_gprofiler_cache_dir(*args, **kwargs) -> Path:

    return get_annotation_dir(*args, **kwargs) / GPROFILER_CACHE_SUBDIR


def _tidy_enrichment(nested_results: dict, id_col: str) -> pd.DataFrame:
    """Stack {key: {celltype: df}} into one long DataFrame with id_col
    identifying which query each block of rows came from, so downstream code
    can filter/facet with a single load instead of re-globbing per-query
    tables."""
    frames = []
    for outer_key, per_celltype in nested_results.items():
        for celltype, df in per_celltype.items():
            if df is None or df.empty:
                continue
            tagged = df.copy()
            tagged[id_col] = outer_key
            tagged["celltype"] = celltype
            frames.append(tagged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_annotation_results(output_dir: Path,
                           enrichment_results: dict,
                           unique_genes: dict,
                           unique_enrichment_results: dict,
                           axis_unique_genes: dict,
                           axis_enrichment_results: dict,
                           top_genes: dict) -> None:
    """Write the tidy enrichment tables and the gene sets they were computed
    on. ``output_dir`` is typically ``get_annotation_dir(...)``'s return
    value; created if it doesn't exist yet."""
    output_dir.mkdir(parents=True, exist_ok=True)

    enrichment_full = _tidy_enrichment(enrichment_results, id_col="method")
    if not enrichment_full.empty:
        enrichment_full["query_type"] = "full_ranking"
    enrichment_full.to_csv(output_dir / ENRICHMENT_FULL_FILENAME, index=False)

    enrichment_unique = _tidy_enrichment(unique_enrichment_results, id_col="method")
    if not enrichment_unique.empty:
        enrichment_unique["query_type"] = "method_unique"
    enrichment_unique.to_csv(output_dir / ENRICHMENT_UNIQUE_FILENAME, index=False)

    enrichment_axis = _tidy_enrichment(axis_enrichment_results, id_col="axis")
    if not enrichment_axis.empty:
        enrichment_axis["query_type"] = "axis_unique"
    enrichment_axis.to_csv(output_dir / ENRICHMENT_AXIS_FILENAME, index=False)

    # Gene sets themselves (not just the enrichment on them), so downstream
    # code can e.g. report set sizes or cross-reference genes without
    # recomputing the set differences.
    with open(output_dir / GENE_SETS_FILENAME, "w") as f:
        json.dump({
            "top_genes": top_genes,
            "method_unique_genes": unique_genes,
            "axis_unique_genes": axis_unique_genes,
        }, f, indent=2)

    log.info(f"Saved enrichment tables and gene sets to {output_dir}")


def annotation_results_exist(output_dir: Path) -> bool:
    return all((output_dir / fname).exists() for fname in (
        ENRICHMENT_FULL_FILENAME, ENRICHMENT_UNIQUE_FILENAME,
        ENRICHMENT_AXIS_FILENAME, GENE_SETS_FILENAME))


def load_annotation_results(output_dir: Path) -> dict:
    """Load everything ``save_annotation_results`` wrote, for notebook /
    downstream use -- mirrors scoring_persistence.load_multiple_results and
    gt_persistence.load_ground_truth's role as the read-side counterpart,
    so callers don't need to know the on-disk filenames."""
    return {
        "enrichment_full": pd.read_csv(output_dir / ENRICHMENT_FULL_FILENAME),
        "enrichment_unique": pd.read_csv(output_dir / ENRICHMENT_UNIQUE_FILENAME),
        "enrichment_axis": pd.read_csv(output_dir / ENRICHMENT_AXIS_FILENAME),
        "gene_sets": json.load(open(output_dir / GENE_SETS_FILENAME)),
    }
