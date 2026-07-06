"""
Functional annotation analysis (GO enrichment) for marker gene rankings
produced by the different dependency measures (Spearman, partial correlation,
mutual information, Integrated Gradients on MLP).

"""

import os
import re
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from gprofiler import GProfiler

PLOTS_DIR = "annotation_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def safe_filename(name):
    """Turn an arbitrary title into a safe, filesystem-friendly filename."""
    name = name.strip().lower()
    name = re.sub(r"[^\w\-]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


methods = {
    "spearman": "results/spearman_results.csv",
    "partial_corr": "results/partial_corr_results.csv",
    # "mi": "results/mi_results.csv",
    "ig_mlp": "results/ig_mlp_results.csv",
}

TOP_N = 50
GO_SOURCES = ["GO:BP"]  # add "GO:MF", "KEGG", "REAC" if needed
SLEEP_BETWEEN_CALLS = 0.5  # seconds, avoid gProfiler rate limiting

active_methods = list(methods.keys())


#Load rankings and extract top-N genes per cell type per method
top_genes = {}  # top_genes[method][celltype] = [gene1, gene2, ...]

for method, path in methods.items():
    df = pd.read_csv(path, index_col="gene_name")
    top_genes[method] = {
        celltype: df[celltype].sort_values(ascending=False).head(TOP_N).index.tolist()
        for celltype in df.columns
    }

celltypes = list(next(iter(top_genes.values())).keys())


# GO enrichment per method / cell type
gp = GProfiler(return_dataframe=True)

enrichment_results = {}  # enrichment_results[method][celltype] = DataFrame

for method in active_methods:
    enrichment_results[method] = {}
    for celltype in celltypes:
        genes = top_genes[method][celltype]
        if len(genes) == 0:
            continue
        try:
            result = gp.profile(
                organism="hsapiens",
                query=genes,
                sources=GO_SOURCES,
            )
        except Exception as e:
            print(f"gProfiler call failed for {method} / {celltype}: {e}")
            continue
        enrichment_results[method][celltype] = result
        time.sleep(SLEEP_BETWEEN_CALLS)


# Method-specific gene sets (set differences)
def genes_unique_to(method, other_methods, celltype):
    others = set()
    for m in other_methods:
        others |= set(top_genes[m][celltype])
    return list(set(top_genes[method][celltype]) - others)


unique_genes = {
    method: {
        celltype: genes_unique_to(
            method, [m for m in active_methods if m != method], celltype
        )
        for celltype in celltypes
    }
    for method in active_methods
}

# Optional: run enrichment specifically on the method-unique gene sets too,
# to see whether "only found by this method" genes are functionally distinct.
unique_enrichment_results = {}

for method in active_methods:
    unique_enrichment_results[method] = {}
    for celltype in celltypes:
        genes = unique_genes[method][celltype]
        if len(genes) == 0:
            continue
        try:
            result = gp.profile(
                organism="hsapiens",
                query=genes,
                sources=GO_SOURCES,
            )
        except Exception as e:
            print(f"gProfiler call failed for {method}-unique / {celltype}: {e}")
            continue
        unique_enrichment_results[method][celltype] = result
        time.sleep(SLEEP_BETWEEN_CALLS)

# 5. Plotting
MAX_TERM_SIZE = 300  # exclude very generic GO terms (e.g. "developmental process")


def plot_top_go_terms(result_df, title, top_k=10, save=True, show=False, max_term_size=None):
    if result_df is None or result_df.empty:
        print(f"No enrichment results to plot for: {title}")
        return

    df = result_df.copy()
    if max_term_size is not None and "term_size" in df.columns:
        before = len(df)
        df = df[df["term_size"] <= max_term_size]
        dropped = before - len(df)
        if dropped:
            print(f"{title}: filtered out {dropped} overly generic GO term(s) "
                  f"(term_size > {max_term_size})")
        if df.empty:
            print(f"No enrichment results left to plot for: {title} "
                  f"after filtering by term_size <= {max_term_size}")
            return

    top = df.nsmallest(top_k, "p_value").copy()
    top["neg_log_p"] = top["p_value"].apply(lambda p: -math.log10(p) if p > 0 else 0)

    plt.figure(figsize=(8, 5))
    bars = plt.barh(top["name"], top["neg_log_p"])
    plt.xlabel("-log10(p-value)")
    plt.title(title)
    plt.gca().invert_yaxis()

    # Annotate each bar with the number of input genes that mapped to that GO term
    if "intersection_size" in top.columns:
        for bar, n_genes in zip(bars, top["intersection_size"]):
            plt.text(
                bar.get_width() + 0.02 * top["neg_log_p"].max(),
                bar.get_y() + bar.get_height() / 2,
                f"n={int(n_genes)}",
                va="center",
                ha="left",
                fontsize=9,
            )
        # extra room on the right so the n= labels aren't cut off
        plt.xlim(0, top["neg_log_p"].max() * 1.15)

    plt.tight_layout()

    if save:
        filepath = os.path.join(PLOTS_DIR, f"{safe_filename(title)}.png")
        plt.savefig(filepath, dpi=300)
        print(f"Saved plot: {filepath}")

    if show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Example: inspect one cell type across all methods
    example_celltype = celltypes[0]
    for method in active_methods:
        plot_top_go_terms(
            enrichment_results[method].get(example_celltype),
            title=f"{method} – {example_celltype}",
        )
    '''
    # Example: inspect method-unique genes for one cell type
    for method in active_methods:
        plot_top_go_terms(
            unique_enrichment_results[method].get(example_celltype),
            title=f"{method} (unique genes) – {example_celltype}",
        )
    '''