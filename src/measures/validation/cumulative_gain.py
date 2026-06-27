import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from networkx.algorithms.bipartite.basic import color
from sklearn.metrics import roc_curve, auc


def prepare_data(results: list[pd.DataFrame],
                 names: list[str],
                 ground_truth):
    def check_marker(row):
        cell_type = row['cell_type']
        true_markers = ground_truth.get(cell_type, [])
        return 1 if row['gene_name'] in true_markers else 0

    plot_data = {}
    for results_df, name in zip(results, names):
        # Melt the results to a long dataframe: move the different cell type values
        # from the columns to a new column
        long_df = results_df.reset_index().melt(
            id_vars='gene_name',
            var_name='cell_type',
            value_name='score'
        )

        # Add column: is_true_marker based on the ground truth
        long_df['is_true_marker'] = long_df.apply(check_marker, axis=1)

        df_sorted = long_df.sort_values(by='score',
                                        ascending=False).reset_index(drop=True)

        y_true = df_sorted['is_true_marker'].values

        n_targets = np.sum(y_true)
        n_samples = len(y_true)
        percentage_samples = np.linspace(0, 100, n_samples + 1)

        cumulative_gains = np.cumsum(y_true)
        percentage_gains = np.append([0], (cumulative_gains / n_targets) * 100)

        # This creates an array of zeros of length 101 (from index 0 to 100).
        # The +1 is there because the chart has to start at (0,0).
        perfect_gains = np.zeros(n_samples + 1)

        # A Perfect model finds a true marker at every single step.
        # the count of found markers goes up by exactly 1 for the first n_targets steps.
        perfect_gains[:n_targets + 1] = np.arange(n_targets + 1)

        # Once index n_targets is reached, all markers have been found.
        # A perfect model has no more markers left to find for the remaining n_samples steps
        perfect_gains[n_targets + 1:] = n_targets

        # Convert to percentage
        percentage_perfect = (perfect_gains / n_targets) * 100

        plot_data[name] = (percentage_samples, percentage_gains,
                           percentage_perfect)


        print(len(percentage_perfect))
        print(percentage_samples)
    return plot_data


def plot(results: list[pd.DataFrame],
         names: list[str],
         ground_truth: dict[str, set]):
    plot_data = prepare_data(results, names, ground_truth)

    fig, ax = plt.subplots(figsize=(5, 5))

    for name, (percentage_samples,
               percentage_gains,
               percentage_perfect) in plot_data.items():

        ax.plot(percentage_samples, percentage_gains, label=f"{name} Method")
        ax.plot(percentage_samples, percentage_perfect, linestyle=':',
            label=f"Perfect Model ({name})")

    ax.plot([0, 100], [0, 100], color='navy', lw=1.5, linestyle='--',
            label='Baseline (Random)')
    ax.set_xlim((0.0, 100.0))
    ax.set_ylim((0.0, 105.0))
    ax.set_xlabel('% of Genes Screened (Sorted by Score)')
    ax.set_ylabel('% of True Markers Recovered')
    ax.set_title('Cumulative Gain Chart')
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=':')
