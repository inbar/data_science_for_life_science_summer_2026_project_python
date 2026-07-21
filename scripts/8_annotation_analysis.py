#!/usr/bin/env python
"""Functional annotation analysis (GO enrichment) for marker gene rankings
produced by the different dependency measures (Spearman, partial correlation,
mutual information, Integrated Gradients on MLP).

Runs after the ranking/scoring scripts have produced
scores/<level>/<subdir>/<method>_results.csv for the given (level,
subsample_size, test_split_size, seed) -- same split-aware subdir naming as
mlp_diagnostics, since scores are computed on the train split.

# To run the scripts run:
# source setup_environment.sh
# In the project root
"""
import argparse
import re
import time
from pathlib import Path

import pandas as pd
from gprofiler import GProfiler

from src import config
from src import logs
from src.persistence import path_tools
from src.persistence import annotation_results as annotation_persistence

import logging

logs.setup_logging(__file__)
log = logging.getLogger(__file__)

# Silence noisy third-party DEBUG output (urllib3/requests connection logs
# from the gprofiler HTTP calls) without touching our own INFO logging.
for noisy_logger in ("urllib3", "urllib3.connectionpool", "requests"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Method name -> filename stem under the results dir for this (level, subsample, seed).
METHOD_RESULT_FILES = {
    "spearman": "spearman_results.csv",
    "partial_corr": "partial_corr_results.csv",
    "mi": "mi_ksg_results.csv",
    "ig_mlp": "ig_mlp_results.csv",
}

ORDERED_QUERY_CAP = 1000  # cap for the "full ranking" enrichment calls below.
GO_SOURCES = ["GO:BP"]  # add "GO:MF", "KEGG", "REAC" if needed
SLEEP_BETWEEN_CALLS = 0.2  # seconds, avoid gProfiler rate limiting

# 2x2 grid of dependency-measure axes: nonlinear vs. linear, conditional vs. marginal.
AXIS_GROUPS = {
    "nonlinear_specific": (["mi", "ig_mlp"], ["spearman", "partial_corr"]),
    "linear_specific": (["spearman", "partial_corr"], ["mi", "ig_mlp"]),
    "conditional_specific": (["partial_corr", "ig_mlp"], ["spearman", "mi"]),
    "marginal_specific": (["spearman", "mi"], ["partial_corr", "ig_mlp"]),
}


def safe_filename(name):
    """Turn an arbitrary cache key into a safe, filesystem-friendly filename."""
    name = name.strip().lower()
    name = re.sub(r"[^\w\-]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def get_results_dir(subsample_size, level, seed, test_split_size, root_dir):
    subdir = path_tools.get_subfolder_path(
        subsample_size=subsample_size, level=level,
        test_split_size=test_split_size, seed=seed)
    return root_dir / "scores" / subdir


def cached_profile(gp_client, cache_dir, cache_key, **profile_kwargs):
    """Run gp.profile(**profile_kwargs), caching the result to disk under
    cache_key so repeated runs (e.g. after a crash) don't repeat slow,
    rate-limited network calls.

    NOTE: cache_key must encode every parameter that affects the result
    (top_n, ORDERED_QUERY_CAP, background gene set, ...), not just
    method/celltype/query-type -- otherwise a rerun with different
    parameters silently returns a stale cached result instead of hitting
    gProfiler again.
    """
    cache_path = cache_dir / f"{safe_filename(cache_key)}.pkl"
    if cache_path.exists():
        return pd.read_pickle(cache_path)
    result = gp_client.profile(**profile_kwargs)
    result.to_pickle(cache_path)
    return result


def load_rankings(results_dir, top_n):
    """Load rankings and extract gene rankings per cell type per method.

    full_ranked_genes: complete gene ranking (best to worst) -> used for
                        enrichment with ordered_query=True, so no arbitrary
                        cutoff is imposed.
    top_genes: top-N cutoff -> only used to define discrete "found by this
               method" sets for the set-difference (unique genes / axis)
               analysis below.
    background_genes: union, across all methods, of every gene that was
                       actually scored -- i.e. the real measured gene panel,
                       not the whole human genome. Passed to gProfiler as a
                       custom statistical background (see run_enrichment
                       etc.) so enrichment p-values reflect "enriched
                       relative to what we could possibly have found" rather
                       than "enriched relative to the ~20k-gene genome",
                       which would bias every result given this dataset only
                       measures ~2k genes.
    """
    full_ranked_genes = {}
    top_genes = {}
    background_genes = set()
    for method, filename in METHOD_RESULT_FILES.items():
        path = results_dir / filename
        df = pd.read_csv(path, index_col="gene_name")
        background_genes |= set(df.index)
        full_ranked_genes[method] = {
            celltype: df[celltype].sort_values(ascending=False).index.tolist()
            for celltype in df.columns
        }
        top_genes[method] = {
            celltype: ranked[:top_n]
            for celltype, ranked in full_ranked_genes[method].items()
        }
    return full_ranked_genes, top_genes, sorted(background_genes)


def genes_unique_to(top_genes, method, other_methods, celltype):
    others = set()
    for m in other_methods:
        others |= set(top_genes[m][celltype])
    return list(set(top_genes[method][celltype]) - others)


def genes_unique_to_group(top_genes, group_methods, other_methods, celltype):
    group_genes = set()
    for m in group_methods:
        group_genes |= set(top_genes[m][celltype])
    other_genes = set()
    for m in other_methods:
        other_genes |= set(top_genes[m][celltype])
    return list(group_genes - other_genes)


def run_enrichment(gp, cache_dir, active_methods, celltypes, full_ranked_genes,
                    ordered_query_cap, background_genes):
    """GO enrichment per method / cell type, on the (capped) full ranking."""
    enrichment_results = {}
    for method in active_methods:
        enrichment_results[method] = {}
        for celltype in celltypes:
            genes = full_ranked_genes[method][celltype][:ordered_query_cap]
            if len(genes) == 0:
                continue
            try:
                result = cached_profile(
                    gp, cache_dir,
                    cache_key=f"{method}__{celltype}__ordered__cap{ordered_query_cap}",
                    organism="hsapiens",
                    query=genes,
                    ordered=True,  # use the (capped) ranking instead of an
                                   # arbitrary top-N cutoff
                    sources=GO_SOURCES,
                    background=background_genes,
                    domain_scope="custom",
                )
            except Exception as e:
                log.warning(f"gProfiler call failed for {method} / {celltype}: {e}")
                continue
            enrichment_results[method][celltype] = result
            time.sleep(SLEEP_BETWEEN_CALLS)
    return enrichment_results


def run_unique_enrichment(gp, cache_dir, active_methods, celltypes, top_genes,
                          top_n, background_genes):
    """Enrichment on method-unique gene sets, to see whether "only found by
    this method" genes are functionally distinct."""
    unique_genes = {
        method: {
            celltype: genes_unique_to(
                top_genes, method, [m for m in active_methods if m != method], celltype
            )
            for celltype in celltypes
        }
        for method in active_methods
    }

    unique_enrichment_results = {}
    for method in active_methods:
        unique_enrichment_results[method] = {}
        for celltype in celltypes:
            genes = unique_genes[method][celltype]
            if len(genes) == 0:
                continue
            try:
                result = cached_profile(
                    gp, cache_dir,
                    cache_key=f"{method}__{celltype}__unique__topn{top_n}",
                    organism="hsapiens",
                    query=genes,
                    sources=GO_SOURCES,
                    background=background_genes,
                    domain_scope="custom",
                )
            except Exception as e:
                log.warning(f"gProfiler call failed for {method}-unique / {celltype}: {e}")
                continue
            unique_enrichment_results[method][celltype] = result
            time.sleep(SLEEP_BETWEEN_CALLS)
    return unique_genes, unique_enrichment_results


def run_axis_enrichment(gp, cache_dir, celltypes, top_genes, top_n, background_genes):
    """GO enrichment on the axis-specific gene sets, e.g. "genes only the
    nonlinear methods (MI, IG) pick up, that neither linear method
    (Spearman, partial corr) picks up" vs. the mirror image for the
    conditional axis."""
    axis_unique_genes = {
        axis_name: {
            celltype: genes_unique_to_group(top_genes, group_methods, other_methods, celltype)
            for celltype in celltypes
        }
        for axis_name, (group_methods, other_methods) in AXIS_GROUPS.items()
    }

    axis_enrichment_results = {}
    for axis_name in AXIS_GROUPS:
        axis_enrichment_results[axis_name] = {}
        for celltype in celltypes:
            genes = axis_unique_genes[axis_name][celltype]
            if len(genes) == 0:
                continue
            try:
                result = cached_profile(
                    gp, cache_dir,
                    cache_key=f"{axis_name}__{celltype}__axis__topn{top_n}",
                    organism="hsapiens",
                    query=genes,
                    sources=GO_SOURCES,
                    background=background_genes,
                    domain_scope="custom",
                )
            except Exception as e:
                log.warning(f"gProfiler call failed for {axis_name} / {celltype}: {e}")
                continue
            axis_enrichment_results[axis_name][celltype] = result
            time.sleep(SLEEP_BETWEEN_CALLS)
    return axis_unique_genes, axis_enrichment_results


def main(args):
    level = args.level
    subsample_size = args.subsample_size
    seed = args.seed
    test_split_size = args.test_split_size
    root_dir = args.root_dir
    top_n = args.top_n

    log.info("Annotation analysis")
    log.info("=====================")
    for k, v in vars(args).items():
        log.info(f"   {k}: {v}")
    log.info("")

    results_dir = get_results_dir(subsample_size, level, seed, test_split_size, root_dir)
    output_dir = annotation_persistence.get_annotation_dir(
        subsample_size=subsample_size, level=level,
        test_split_size=test_split_size, seed=seed, root_dir=root_dir)
    cache_dir = annotation_persistence.get_gprofiler_cache_dir(
        subsample_size=subsample_size, level=level,
        test_split_size=test_split_size, seed=seed, root_dir=root_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    active_methods = list(METHOD_RESULT_FILES.keys())
    full_ranked_genes, top_genes, background_genes = load_rankings(results_dir, top_n)
    celltypes = list(next(iter(top_genes.values())).keys())
    log.info(f"Background gene set size (custom domain_scope): {len(background_genes)}")

    gp = GProfiler(return_dataframe=True)

    enrichment_results = run_enrichment(
        gp, cache_dir, active_methods, celltypes, full_ranked_genes,
        ORDERED_QUERY_CAP, background_genes)

    unique_genes, unique_enrichment_results = run_unique_enrichment(
        gp, cache_dir, active_methods, celltypes, top_genes, top_n, background_genes)

    axis_unique_genes, axis_enrichment_results = run_axis_enrichment(
        gp, cache_dir, celltypes, top_genes, top_n, background_genes)

    annotation_persistence.save_annotation_results(
        output_dir,
        enrichment_results, unique_genes, unique_enrichment_results,
        axis_unique_genes, axis_enrichment_results, top_genes,
    )

    log.info(f"Done. Artifacts written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default=config.DEFAULT_LEVEL)
    parser.add_argument("--subsample_size", type=int,
                        default=config.DEFAULT_SUBSAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--test_split_size", type=int, required=True,
                        help="Same test_split_size used for the train/test split "
                             "the scores were computed on (scores/<level>/<subdir> "
                             "is split-aware, like mlp_diagnostics -- see "
                             "src.persistence.path_tools.get_subfolder_path).")
    parser.add_argument("--root_dir", type=str, default=str(config.LOCAL_DATA_ROOT),
                        help="Repo-local results root (annotation artifacts are "
                             "meant to be inspected from the repo, unlike the "
                             "large cached datasets under PROCESSED_DATA). Also "
                             "where the ranking scores/<level>/<subdir>/*.csv "
                             "files are expected to already exist.")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Top-N cutoff used only for the method-unique / "
                             "axis-unique set-difference analyses; the ordered "
                             "full-ranking enrichment ignores this.")
    parsed_args = parser.parse_args()
    parsed_args.root_dir = Path(parsed_args.root_dir)

    main(parsed_args)
