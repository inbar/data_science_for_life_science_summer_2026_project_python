"""Publication-quality plotting helpers.

House style (enforced by :func:`set_style`): no titles, no chart-junk, minimal
spines, colour-blind-safe palette, embedded fonts. Every figure is saved by
:func:`save` as both a vector PDF and a 600 dpi PNG into ``results/figures``.

The notebook only ever calls these one-liners; all layout logic lives here.
"""
from __future__ import annotations

from itertools import combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from sympy import false, true

from config import FIGURES_DIR_PATH
from config import METHOD_LABELS

# A colour-blind-safe qualitative palette (Wong 2011).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]
DRIVER_COLOR = "#D55E00"
BG_COLOR = "#BBBBBB"


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,  # embed TrueType (editable)
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "figure.constrained_layout.use": True,
    })


def get_or_create_figure_dir(name, level=None):
    """Directory figures are written to for ``fmt`` at the active level."""
    dir = (FIGURES_DIR_PATH / level / name) if lvl else FIGURES_DIR_PATH
    dir.mkdir(parents=True, exist_ok=True)
    return dir


def save(fig, name, level=None, tight=True):
    """Save as vector PDF + 600 dpi PNG (no title), into the level/format dirs set
    by :func:`set_fig_level`. ``tight=False`` keeps the figure's own margins
    (needed for 3D axes, whose z-label a tight crop drops)."""
    kw = {} if tight else {"bbox_inches": None}
    pdf_dir = get_or_create_figure_dir("pdf", level)
    png_dir = get_or_create_figure_dir("png", level)
    fig.savefig(pdf_dir / f"{name}.pdf", **kw)
    fig.savefig(png_dir / f"{name}.png", dpi=600, **kw)


def _mlabel(m: str) -> str:
    return METHOD_LABELS.get(m, m)


# --------------------------------------------------------------------------- #
# Exploratory / QC
# --------------------------------------------------------------------------- #
def qc_violins(qc_metrics: pd.DataFrame, cols, figsize=None):
    """Violin+strip of QC metrics (one panel each). ``qc`` rows = cells."""
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=figsize or (1.8 * n, 2.2))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        value = qc_metrics[col].values
        parts = ax.violinplot(value, showextrema=false)
        for b in parts["bodies"]:
            b.set_facecolor(PALETTE[0])
            b.set_alpha(0.6)
        ax.boxplot(value, widths=0.15,
                   showfliers=False,
                   medianprops=dict(color="k", lw=1))
        ax.set_ylabel(col)
    return fig


def pca_variance_ratio(dataset):
    sc.pl.pca_variance_ratio(dataset)


def plot_embedding(dataset,
                   obsm_key,
                   title=None,
                   color_by_key=None,
                   legend_loc="right margin",
                   figsize=(3, 3),
                   ax=None,
                   show=False):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        show = True

    sc.pl.embedding(dataset,
                    basis=obsm_key,
                    title=title,
                    color=color_by_key,
                    palette="inferno",
                    size=8,
                    legend_loc=legend_loc,
                    legend_fontsize=8,
                    ax=ax,
                    show=show)


def embedding_scatter(dataset,
                      labels,
                      palette=None,
                      s=2,
                      alpha=0.7,
                      legend=True,
                      figsize=(3.4, 3.0),
                      order=None,
                      legend_ncol=1,
                      rasterized=True):
    """Generic 2D embedding (UMAP/PCA) coloured by a categorical label.

    No title; axes labelled UMAP1/UMAP2 by the caller via ``ax`` return.
    """
    labels = np.asarray(labels).astype(str)
    cats = order if order is not None else sorted(pd.unique(labels))
    pal = palette or {c: PALETTE[i % len(PALETTE)] for i, c in
                      enumerate(cats)}
    fig, ax = plt.subplots(figsize=figsize)
    for c in cats:
        m = labels == c
        ax.scatter(dataset[m, 0], dataset[m, 1], s=s, alpha=alpha,
                   c=pal[c], label=c, linewidths=0, rasterized=rasterized)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1");
    ax.set_ylabel("UMAP 2")
    if legend:
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4,
                          markerfacecolor=pal[c], markeredgecolor="none",
                          label=c)
                   for c in cats]
        ax.legend(handles=handles, loc="center left",
                  bbox_to_anchor=(1.0, 0.5),
                  ncol=legend_ncol, handletextpad=0.2, labelspacing=0.25)
    return fig, ax


def protein_marker_validation_heatmap(dataframe):
    """Cell-type (rows) x protein (cols) heatmap for label validation."""

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(dataframe,
                   aspect="auto",
                   cmap="viridis",
                   vmin=-2.5,
                   vmax=2.5)

    x_vals = dataframe.columns.unique()
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels(x_vals, rotation=45)

    ax.set_yticks(np.arange(len(dataframe.index)))
    ax.set_yticklabels(dataframe.index)

    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Z-score (CLR expression)")


# --------------------------------------------------------------------------- #
# Per-method diagnostics
# --------------------------------------------------------------------------- #
def mlp_training_curves(history, figsize=(5.2, 2.2)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ep = range(1, len(history.train_loss) + 1)
    ax1.plot(ep, history.train_loss, color=PALETTE[0], label="train")
    ax1.plot(ep, history.val_loss, color=PALETTE[1], label="validation")
    ax1.axvline(history.best_epoch + 1, color="k", ls="--", lw=0.8)
    ax1.set_xlabel("epoch");
    ax1.set_ylabel("cross-entropy loss");
    ax1.legend()
    ax2.plot(ep, history.val_acc, color=PALETTE[2])
    ax2.axvline(history.best_epoch + 1, color="k", ls="--", lw=0.8)
    ax2.set_xlabel("epoch");
    ax2.set_ylabel("validation accuracy")
    return fig, (ax1, ax2)


def per_class_bar(series_map: dict, xlabel="F1 score", figsize=None):
    """Grouped horizontal bars of a per-class metric for several models.

    ``series_map`` maps a model label -> pandas Series indexed by class. Classes
    are ordered by the first model's values (worst at bottom)."""
    labels = list(series_map)
    order = series_map[labels[0]].index.tolist()
    classes = order
    y = np.arange(len(classes))
    h = 0.8 / len(labels)
    fig, ax = plt.subplots(
        figsize=figsize or (3.6, 0.22 * len(classes) + 1.0))
    for i, lab in enumerate(labels):
        vals = series_map[lab].reindex(classes).values
        ax.barh(y + i * h, vals, height=h, color=PALETTE[i % len(PALETTE)],
                label=lab, edgecolor="none")
    ax.set_yticks(y + h * (len(labels) - 1) / 2)
    ax.set_yticklabels(classes)
    ax.set_xlabel(xlabel);
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    return fig, ax


def confusion_heatmap(cm: np.ndarray, classes, figsize=None,
                      normalize=True):
    """Row-normalised confusion matrix heatmap (true = rows, predicted = cols)."""
    M = cm.astype(float)
    if normalize:
        M = M / np.clip(M.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=figsize or (0.28 * len(classes) + 1.8,
                                               0.28 * len(classes) + 1.5))
    im = ax.imshow(M, cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)));
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(len(classes)));
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted");
    ax.set_ylabel("true")
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02);
    cb.set_label("fraction")
    return fig, ax


def score_hist(values, xlabel, figsize=(3.0, 2.2), bins=50, color=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(np.asarray(values), bins=bins, color=color or PALETTE[0],
            alpha=0.85)
    ax.set_xlabel(xlabel);
    ax.set_ylabel("genes")
    return fig, ax


def eigen_spectrum(eigs_raw, eigs_shrunk, figsize=(3.2, 2.4)):
    """Covariance eigenvalue spectra: raw vs Ledoit-Wolf shrinkage (log y)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.sort(eigs_raw)[::-1], color=PALETTE[1], label="empirical")
    ax.plot(np.sort(eigs_shrunk)[::-1], color=PALETTE[0],
            label="Ledoit-Wolf")
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue rank");
    ax.set_ylabel("eigenvalue")
    ax.legend()
    return fig, ax


# --------------------------------------------------------------------------- #
# Benchmark results
# --------------------------------------------------------------------------- #
def auc_box(df: pd.DataFrame, method_order, sig_pairs=None,
            figsize=(3.6, 2.8)):
    """Box + strip of AUC_rel per method (one point per cell type)."""
    fig, ax = plt.subplots(figsize=figsize)
    data = [df.loc[df.method == m, "auc_rel"].dropna().values for m in
            method_order]
    bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color="k", lw=1))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(PALETTE[i % len(PALETTE)]);
        box.set_alpha(0.35)
        box.set_edgecolor("k")
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        x = rng.normal(i + 1, 0.06, size=len(d))
        ax.scatter(x, d, s=6, color=PALETTE[i % len(PALETTE)], alpha=0.9,
                   linewidths=0, zorder=3)
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.set_xticks(range(1, len(method_order) + 1))
    ax.set_xticklabels([_mlabel(m) for m in method_order], rotation=25,
                       ha="right")
    ax.set_ylabel(r"driver-recovery AUC$_{\mathrm{rel}}$")

    if sig_pairs:
        _annotate_sig(ax, method_order, sig_pairs, data)
    return fig, ax


def _annotate_sig(ax, method_order, sig_pairs, data):
    top = max(np.nanmax(d) for d in data if len(d))
    step = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0] + 1)
    idx = {m: i + 1 for i, m in enumerate(method_order)}
    lvl = 0
    for (a, b, label) in sig_pairs:
        if a not in idx or b not in idx:
            continue
        x1, x2 = idx[a], idx[b]
        y = top + step * (lvl + 1)
        ax.plot([x1, x1, x2, x2], [y, y + step * 0.3, y + step * 0.3, y],
                color="k", lw=0.8)
        ax.text((x1 + x2) / 2, y + step * 0.3, label, ha="center",
                va="bottom",
                fontsize=6)
        lvl += 1


def recovery_curves(curves: dict, figsize=(3.2, 2.8)):
    """``curves`` maps method -> (frac_genes, frac_recovered)."""
    fig, ax = plt.subplots(figsize=figsize)
    for i, (m, (x, y)) in enumerate(curves.items()):
        ax.plot(x, y, color=PALETTE[i % len(PALETTE)], label=_mlabel(m))
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.8)
    ax.set_xlabel("fraction of ranked genes")
    ax.set_ylabel("fraction of drivers recovered")
    ax.legend(loc="lower right")
    return fig, ax


def recovery_at_k_curve(df: pd.DataFrame, methods, exclude=None,
                        n_boot=2000,
                        seed=0, figsize=(3.4, 2.8)):
    """Mean recall@k vs k per method, with a bootstrap-over-cell-types CI band.

    ``df`` is the tidy [celltype, method, k, recovery] table. Cell type is the
    replication unit, so the CI is obtained by resampling cell types. ``exclude``
    drops named cell types (e.g. heterogeneous 'other' categories)."""
    if exclude:
        df = df[~df["celltype"].isin(exclude)]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=figsize)
    for i, m in enumerate(methods):
        piv = df[df.method == m].pivot_table(index="celltype", columns="k",
                                             values="recovery")
        ks = piv.columns.values.astype(float)
        vals = piv.values  # (cell types x k)
        mean = np.nanmean(vals, axis=0)
        nct = vals.shape[0]
        boot = np.array([np.nanmean(vals[rng.integers(0, nct, nct)], axis=0)
                         for _ in range(n_boot)])
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        ax.plot(ks, mean, color=PALETTE[i % len(PALETTE)], label=_mlabel(m))
        ax.fill_between(ks, lo, hi, color=PALETTE[i % len(PALETTE)],
                        alpha=0.16,
                        linewidth=0)
    ax.set_xlabel("top-k genes");
    ax.set_ylabel("fraction of drivers recovered")
    ax.legend(loc="upper left")
    return fig, ax


def recovery_curves_panel(panel: dict, ncols=4, xlim=None, figsize=None):
    """Grid of cumulative recovery curves, one panel per cell type.

    ``panel`` maps a cell-type label -> {method: (frac_genes, frac_recovered)}.
    x is the fraction of *all* ranked genes; pass ``xlim=(0, f)`` to zoom into the
    early region. The cell type is annotated inside each panel (no titles)."""
    n = len(panel)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharey=True, sharex=True,
                             figsize=figsize or (2.0 * ncols, 1.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (ct, curves) in zip(axes, panel.items()):
        for i, (m, (x, y)) in enumerate(curves.items()):
            ax.plot(x, y, color=PALETTE[i % len(PALETTE)], label=_mlabel(m))
        if xlim is None:
            ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.8)
        else:
            ax.set_xlim(*xlim)
        ax.text(0.05, 0.97, ct, transform=ax.transAxes, va="top", ha="left",
                fontsize=6.5, fontweight="bold")
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n - ncols):n]:
        ax.set_xlabel("fraction of ranked genes")
    for r in range(nrows):
        axes[r * ncols].set_ylabel("drivers recovered")
    axes[ncols - 1].legend(loc="lower right", fontsize=5.5)
    return fig, axes


def recovery_curve_mean(store: dict, methods, exclude=None, n_boot=2000,
                        seed=0, xmax=1.0, figsize=(3.4, 2.8)):
    """Cumulative recovery curve **averaged over cell types** (bootstrap-over-cell
    -type band). x is the fraction of all ranked genes; ``xmax`` < 1 zooms into the
    early region. ``store`` is the per-cell-type score dict from the benchmark."""
    from .metric import recovery_curve
    cts = [c for c in store if not (exclude and c in exclude)]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=figsize)
    x_ref = None
    for i, m in enumerate(methods):
        ys = []
        for ct in cts:
            if m not in store[ct]:
                continue
            x, y = recovery_curve(store[ct][m], store[ct]["_driver_mask"])
            ys.append(y);
            x_ref = x
        Y = np.vstack(ys)  # (cell types x genes)
        mean = Y.mean(0)
        boot = np.array([Y[rng.integers(0, Y.shape[0], Y.shape[0])].mean(0)
                         for _ in range(n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        ax.plot(x_ref, mean, color=PALETTE[i % len(PALETTE)],
                label=_mlabel(m))
        ax.fill_between(x_ref, lo, hi, color=PALETTE[i % len(PALETTE)],
                        alpha=0.16, linewidth=0)
    if xmax >= 1.0:
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.8)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("fraction of ranked genes")
    ax.set_ylabel("fraction of drivers recovered")
    ax.legend(loc="lower right")
    return fig, ax


def auc_heatmap(wide: pd.DataFrame, method_order=None, figsize=None):
    """Cell-type (rows) x method (cols) AUC_rel heatmap."""
    cols = method_order or list(wide.columns)
    w = wide[cols]
    fig, ax = plt.subplots(figsize=figsize or (0.7 * len(cols) + 2.0,
                                               0.22 * w.shape[0] + 1.0))
    im = ax.imshow(w.values, aspect="auto", cmap="viridis", vmin=0.5,
                   vmax=1.0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([_mlabel(m) for m in cols], rotation=25, ha="right")
    ax.set_yticks(range(w.shape[0]));
    ax.set_yticklabels(w.index)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(r"AUC$_{\mathrm{rel}}$")
    return fig, ax


# --------------------------------------------------------------------------- #
# Cross-method comparison (gene-level)
# --------------------------------------------------------------------------- #
def pairwise_scatter_matrix(long: pd.DataFrame, methods, figsize=(6.4, 6.4),
                            s=3, point_alpha=0.4):
    """Lower-triangle scatter matrix of per-gene method scores.

    ``long`` has one row per (gene, cell type) with a column per method and a
    boolean ``is_driver``. Drivers are drawn on top in the highlight colour.
    Diagonal shows the score distribution; upper triangle shows Spearman rho.
    """
    k = len(methods)
    fig, axes = plt.subplots(k, k, figsize=figsize)
    drv = long["is_driver"].values
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            ax = axes[i, j]
            if i == j:
                ax.hist(long[mi].values, bins=40, color=BG_COLOR)
                ax.set_yticks([])
            elif i > j:
                ax.scatter(long[mj].values[~drv], long[mi].values[~drv],
                           s=s, c=BG_COLOR, alpha=point_alpha, linewidths=0,
                           rasterized=True)
                ax.scatter(long[mj].values[drv], long[mi].values[drv],
                           s=s + 4, c=DRIVER_COLOR, alpha=0.9, linewidths=0,
                           rasterized=True, zorder=3)
            else:
                rho = pd.Series(long[mi].values).corr(
                    pd.Series(long[mj].values), method="spearman")
                ax.text(0.5, 0.5, f"ρ = {rho:.2f}", ha="center",
                        va="center",
                        transform=ax.transAxes, fontsize=8)
                ax.set_xticks([]);
                ax.set_yticks([])
            if i == k - 1:
                ax.set_xlabel(_mlabel(mj), fontsize=6)
            if j == 0:
                ax.set_ylabel(_mlabel(mi), fontsize=6)
            ax.tick_params(labelsize=5)
    # one shared legend
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4,
                      markerfacecolor=DRIVER_COLOR, markeredgecolor="none",
                      label="protein-derived driver"),
               Line2D([0], [0], marker="o", linestyle="", markersize=4,
                      markerfacecolor=BG_COLOR, markeredgecolor="none",
                      label="other gene")]
    fig.legend(handles=handles, loc="upper right", ncol=1)
    return fig, axes


def scatter3d(long: pd.DataFrame, x, y, z, elev=22, azim=-60,
              figsize=(5.6, 4.2),
              s=4):
    """3D scatter of three method scores, drivers highlighted."""
    drv = long["is_driver"].values
    # constrained_layout does not support 3D axes well and clips the z-label;
    # save this figure with tight=False so the fixed margins below are kept.
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(long[x].values[~drv], long[y].values[~drv],
               long[z].values[~drv],
               s=s, c=BG_COLOR, alpha=0.25, linewidths=0, depthshade=False)
    ax.scatter(long[x].values[drv], long[y].values[drv],
               long[z].values[drv],
               s=s + 10, c=DRIVER_COLOR, alpha=0.95, linewidths=0,
               depthshade=False)
    ax.set_xlabel(_mlabel(x), labelpad=8)
    ax.set_ylabel(_mlabel(y), labelpad=8)
    ax.view_init(elev=elev, azim=azim)
    fig.subplots_adjust(left=-0.02, right=0.84, bottom=0.04, top=0.98)
    # mpl's 3D z-label placement is unreliable across versions / with tight bbox,
    # so the z-axis title is placed explicitly in figure coordinates (right edge).
    fig.text(0.965, 0.5, _mlabel(z), rotation=90, va="center", ha="center",
             fontsize=8)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4,
                      markerfacecolor=DRIVER_COLOR, markeredgecolor="none",
                      label="protein-derived driver"),
               Line2D([0], [0], marker="o", linestyle="", markersize=4,
                      markerfacecolor=BG_COLOR, markeredgecolor="none",
                      label="other gene")]
    ax.legend(handles=handles, loc="upper left")
    return fig, ax
