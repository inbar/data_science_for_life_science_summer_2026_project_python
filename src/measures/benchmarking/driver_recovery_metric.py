import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn


def to_long_df(df: pd.DataFrame,
               ground_truth):
    def check_marker(row):
        cell_type = row['cell_type']
        true_markers = ground_truth.get(cell_type, [])
        return 1 if row['gene_name'] in true_markers else 0

    long_df = df.reset_index().melt(
        id_vars='gene_name',
        var_name='cell_type',
        value_name='score'
    )

    # Add column: is_true_marker based on the ground truth
    long_df['is_true_marker'] = long_df.apply(check_marker, axis=1)

    return long_df


def calc(results: pd.DataFrame,
         ground_truth):
    long_df = to_long_df(results, ground_truth)

    df_sorted = long_df.sort_values(by='score',
                                    ascending=False).reset_index(drop=True)

    y_true = df_sorted['is_true_marker'].values

    n_targets = np.sum(y_true)
    n_samples = len(y_true)

    # Define the X-axis (samples screened)
    # Starts at 0, ends at n_samples
    samples_considered = np.arange(n_samples + 1)

    # Construct Y-axis (cumulative true targets found)
    cumulative_true_targets_found = np.append([0], np.cumsum(y_true))

    # A Perfect model finds a true marker at every single step.
    # the count of found markers goes up by exactly 1 for the first n_targets steps.
    cumulative_true_targets_found_perfect = np.zeros(n_samples + 1)
    cumulative_true_targets_found_perfect[:n_targets + 1] = np.arange(
        n_targets + 1)
    cumulative_true_targets_found_perfect[n_targets + 1:] = n_targets

    return (samples_considered,
            cumulative_true_targets_found,
            cumulative_true_targets_found_perfect,
            n_targets)


def compute_auc_rel(results: pd.DataFrame,
                    ground_truth):
    (x_coordinates,
     y_coordinates,
     y_coordinates_perfect, _) = calc(results, ground_truth)

    auc_model = sklearn.metrics.auc(x_coordinates,
                                    y_coordinates)
    auc_perfect = sklearn.metrics.auc(x_coordinates,
                                      y_coordinates_perfect)

    return auc_model / auc_perfect


def prepare_plot_data(results: list[pd.DataFrame],
                      names: list[str],
                      ground_truth):
    plot_data = {}
    for results_df, name in zip(results, names):
        (samples_considered,
         cumulative_true_targets_found,
         cumulative_true_targets_found_perfect,
         n_targets) = calc(results_df, ground_truth)

        percent_samples = np.linspace(0, 100, len(samples_considered))
        percent_cumulative = (cumulative_true_targets_found / n_targets) * 100
        percent_cumulative_perfect = (cumulative_true_targets_found_perfect
                                      /
                                      n_targets) * 100

        auc_rel = compute_auc_rel(results_df, ground_truth)

        plot_data[name] = (
            percent_samples,
            percent_cumulative,
            percent_cumulative_perfect,
            auc_rel
        )

    return plot_data


def plot(results: list[pd.DataFrame],
         names: list[str],
         ground_truth: dict[str, set]):
    plot_data = prepare_plot_data(results, names, ground_truth)

    fig, ax = plt.subplots(figsize=(5, 5))

    for name, (percent_samples,
               percent_cumulative,
               percent_cumulative_perfect,
               auc_rel) in plot_data.items():
        ax.plot(percent_samples, percent_cumulative, label=f"{name} Method (AUC = {auc_rel:.2f})")
        ax.plot(percent_samples, percent_cumulative_perfect, linestyle=':',
                label=f"Perfect Model ({name})")
        ax.fill_between(percent_samples, percent_cumulative, alpha=0.1)

    ax.plot([0, 100], [0, 100], color='navy', lw=1.5, linestyle='--',
            label='Baseline (Random)')
    ax.set_xlim((0.0, 100.0))
    ax.set_ylim((0.0, 105.0))
    ax.set_xlabel('% of Genes Screened (Sorted by Score)')
    ax.set_ylabel('% of True Markers Recovered')
    ax.set_title('AUC_rel')
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=':')
