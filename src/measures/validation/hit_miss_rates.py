import pandas as pd
from matplotlib import pyplot as plt

from src.plotting import set_style


def calculate_validation_data(results_df: pd.DataFrame,
                              ground_truth: dict[str, set],
                              top_k=50) -> pd.DataFrame:
    intersection_results = {}
    for cell_type in results_df.columns:
        if cell_type not in ground_truth:
            continue

        driver_genes = ground_truth[cell_type]

        top_k_predictions = results_df[cell_type].sort_values(
            ascending=False).head(top_k).index

        hits = driver_genes.intersection(top_k_predictions)
        misses = driver_genes.difference(top_k_predictions)

        intersection_results[cell_type] = {
            'n_drivers': len(driver_genes),
            'n_hits': len(hits),
            'n_misses': len(misses),
            'hits': hits,
            'misses': misses
        }

    performance_evaluation_df = pd.DataFrame.from_dict(intersection_results,
                                                       orient='index')

    performance_evaluation_df["hit_rate"] = (
        performance_evaluation_df["n_hits"]
        /
        performance_evaluation_df["n_drivers"]
    )

    performance_evaluation_df["miss_rate"] = (
        performance_evaluation_df["n_misses"]
        /
        performance_evaluation_df["n_drivers"]
    )

    rates_df = performance_evaluation_df[["hit_rate", "miss_rate"]]

    summary = {
        "avg": rates_df.mean(axis=0),
        "median": rates_df.median(axis=0)
    }

    summary_df = pd.DataFrame(data=summary,
                              index=["hit_rate", "miss_rate"])

    return summary_df


def plot(summaries: list[pd.DataFrame],
         names: list[str]):
    set_style()

    combined_hit_rates_df = pd.DataFrame()
    combined_miss_rates_df = pd.DataFrame()

    for summary, name in zip(summaries, names):
        combined_hit_rates_df[name] = summary.loc["hit_rate"]
        combined_miss_rates_df[name] = summary.loc["miss_rate"]

    combined_hit_rates_df = combined_hit_rates_df.T
    combined_miss_rates_df = combined_miss_rates_df.T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    ax1.plot(combined_hit_rates_df.index, combined_hit_rates_df['median'],
             marker='s', linestyle='-', color='blue', linewidth=2,
             label='Median')
    ax1.plot(combined_hit_rates_df.index, combined_hit_rates_df['avg'],
             marker='^', linestyle='-', color='red', linewidth=2,
             label='Average')

    ax1.set_title('Hit Rate comparison across methods')
    ax1.set_ylabel('Hit Rate')
    ax1.set_xlabel('Methods', labelpad=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')

    ax2.plot(combined_miss_rates_df.index, combined_miss_rates_df['median'],
             marker='s',
             linestyle='-', color='blue', linewidth=2, label='Median')
    ax2.plot(combined_miss_rates_df.index, combined_miss_rates_df['avg'],
             marker='^',
             linestyle='-', color='red', linewidth=2, label='Average')

    ax2.set_title('Miss Rate comparison across methods')
    ax2.set_ylabel('Miss Rate')
    ax2.set_xlabel('Methods', labelpad=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right')
