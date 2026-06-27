import matplotlib.pyplot as plt
import pandas as pd
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
                                        ascending=False).reset_index(
            drop=True)

        y_true = df_sorted['is_true_marker'].values
        y_scores = df_sorted['score'].values

        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        plot_data[name] = (false_positive_rate, true_positive_rate,
                           roc_auc)

    return plot_data


def plot(results: list[pd.DataFrame],
         names: list[str],
         ground_truth: dict[str, set]):
    plot_data = prepare_data(results, names, ground_truth)

    fig, ax = plt.subplots(figsize=(5, 5))

    for name, (fpr, tpr, roc_auc) in plot_data.items():
        ax.plot(fpr, tpr, label=f'{name} Method (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], linestyle='--', color="navy", label="Random Guess")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC')
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=':')
