import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from scipy import stats
import warnings
import logging
from itertools import combinations
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import os

warnings.filterwarnings("ignore")

from src.config import (
    SYMBOLS,
    INTERVAL,
    START_DATE,
    END_DATE,
    BASE_URL,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def rolling_granger_causality(y, x, max_lag=100, window_size=500):
    """
    Implements time-varying Granger causality test using rolling windows

    Parameters:
    -----------
    y : pd.Series
        Target variable (the variable being predicted)
    x : pd.Series
        Potential causal variable
    max_lag : int
        Maximum number of lags to test
    window_size : int
        Size of the rolling window

    Returns:
    --------
    pd.DataFrame
        DataFrame containing F-statistics and p-values for each window
    """
    if len(y) != len(x):
        raise ValueError("Series must be of equal length")

    # Initialize results storage
    results = []
    dates = []

    # Find optimal lag order using AIC
    # join two series

    # Create lagged versions of both series
    y_lags = pd.DataFrame({f"y_lag_{i}": y.shift(i) for i in range(1, max_lag + 1)})
    x_lags = pd.DataFrame({f"x_lag_{i}": x.shift(i) for i in range(1, max_lag + 1)})

    # Combine all data
    data = pd.concat([y, y_lags, x_lags], axis=1)
    data = data.dropna()

    # Rolling window analysis
    for start in range(0, len(data) - window_size + 1):
        window_data = data.iloc[start : start + window_size]

        # Restricted model (only y lags)
        y_window = window_data.iloc[:, 0]
        y_lags_window = window_data.iloc[:, 1 : max_lag + 1]
        restricted_model = OLS(y_window, sm.add_constant(y_lags_window)).fit()
        rss_restricted = restricted_model.ssr

        # Unrestricted model (y lags and x lags)
        x_lags_window = window_data.iloc[:, max_lag + 1 :]
        full_x = pd.concat([y_lags_window, x_lags_window], axis=1)
        unrestricted_model = OLS(y_window, sm.add_constant(full_x)).fit()
        rss_unrestricted = unrestricted_model.ssr

        # Calculate F-statistic
        n = window_size
        q = max_lag  # number of restrictions
        k = 2 * max_lag  # number of parameters in unrestricted model
        f_stat = ((rss_restricted - rss_unrestricted) / q) / (
            rss_unrestricted / (n - k - 1)
        )
        p_value = 1 - stats.f.cdf(f_stat, q, n - k - 1)

        results.append(
            {
                "date": data.index[start + window_size - 1],
                "f_stat": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }
        )

    return pd.DataFrame(results)


def analyze_tvgc(
    crypto_data, pairs_to_test, window_size=500, test_mode=True, max_lags=100
):
    """
    Analyze time-varying Granger causality for multiple cryptocurrency pairs

    Args:
        crypto_data: Dictionary of cryptocurrency data
        pairs_to_test: List of pairs to test
        window_size: Size of rolling window
        test_mode: If True, only analyze first 1000 data points for quick testing
    """
    tvgc_results = {}
    total_pairs = len(pairs_to_test) * 2  # Both directions
    completed = 0

    for pair in pairs_to_test:
        symbol1, symbol2 = pair
        logger.info(
            f"\nAnalyzing TVGC for {symbol1} -> {symbol2} ({completed}/{total_pairs} pairs completed)"
        )

        # Get returns data
        returns1 = crypto_data[symbol1]
        returns2 = crypto_data[symbol2]

        if test_mode:
            returns1 = returns1[:1000]  # Only use first 1000 points for testing
            returns2 = returns2[:1000]
            logger.info("TEST MODE: Using first 1000 data points only")

        logger.info(f"Processing forward causality: {symbol1} -> {symbol2}")
        forward_results = rolling_granger_causality(
            returns2.dropna(),
            returns1.dropna(),
            window_size=window_size,
            max_lag=max_lags,
        )
        completed += 1
        logger.info(
            f"Forward causality complete. Initial p-value: {forward_results['p_value'].iloc[0]:.4f}"
        )

        logger.info(f"Processing backward causality: {symbol2} -> {symbol1}")
        backward_results = rolling_granger_causality(
            returns1.dropna(),
            returns2.dropna(),
            window_size=window_size,
            max_lag=max_lags,
        )
        completed += 1
        logger.info(
            f"Backward causality complete. Initial p-value: {backward_results['p_value'].iloc[0]:.4f}"
        )

        tvgc_results[f"{symbol1}->{symbol2}"] = forward_results
        tvgc_results[f"{symbol2}->{symbol1}"] = backward_results

        # Print summary of this pair
        print(f"\nSummary for pair {symbol1}/{symbol2}:")
        print(
            f"{symbol1}->{symbol2} avg p-value: {forward_results['p_value'].mean():.4f}"
        )
        print(
            f"{symbol2}->{symbol1} avg p-value: {backward_results['p_value'].mean():.4f}"
        )

    return tvgc_results


def run_tvgc_analysis(
    data: dict[str, pd.DataFrame], window_size: int = 300, test_mode: bool = False
) -> None:
    """Run time-varying Granger causality analysis."""
    # Get all pairs of cryptocurrencies
    symbols = list(data.keys())
    pairs = list(combinations(symbols, 2))

    logger.info(f"Analyzing {len(pairs)} cryptocurrency pairs...")

    # Run TVGC analysis
    tvgc_results = analyze_tvgc(
        data, pairs, window_size=window_size, test_mode=test_mode, max_lags=20
    )

    # Create output directory if it doesn't exist
    output_dir = f"./results/{INTERVAL}/plots"

    # Plot result
    logger.info("Plotting TVGC results...")
    plt.figure(figsize=(15, 10))
    plot_tvgc_results(
        tvgc_results, "Time-Varying Granger Causality Analysis", output_dir
    )

    # Generate and save summary statistics
    print("Generating summary statistics...")
    summary = summarize_tvgc_results(tvgc_results)

    return summary, tvgc_results


def plot_tvgc_results(
    tvgc_results,
    title="Time-Varying Granger Causality Results",
    output_dir="./results/plots",
    figsize=None,
):
    """
    Plot the time-varying Granger causality results

    Parameters:
    -----------
    tvgc_results : dict
        Dictionary containing TVGC results for each pair
    title : str
        Title for the plot
    """

    for pair, results in tvgc_results.items():
        if figsize is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = plt.subplots(figsize=figsize)

        # Plot significance regions
        ax.fill_between(
            results["date"],
            0,
            1,
            where=results["significant"],
            alpha=0.3,
            label=f"{pair} (significant regions)",
        )

        # Plot p-values
        ax.plot(
            results["date"], results["p_value"], label=f"{pair} (p-value)", alpha=0.7
        )

        ax.axhline(y=0.05, color="r", linestyle="--", label="5% significance level")
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("P-value")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()

        # save figure
        fig.savefig(
            os.path.join(output_dir, f"tvgc_results_{'_'.join(pair.split('->'))}.png")
        )
        plt.close(fig)


# Calculate summary statistics for TVGC results
def summarize_tvgc_results(tvgc_results):
    """
    Generate summary statistics for TVGC results

    Parameters:
    -----------
    tvgc_results : dict
        Dictionary containing TVGC results for each pair

    Returns:
    --------
    pd.DataFrame
        Summary statistics for each pair
    """
    summary_data = []

    for pair, results in tvgc_results.items():
        summary = {
            "Pair": pair,
            "Avg p-value": results["p_value"].mean(),
            "Min p-value": results["p_value"].min(),
            "Max p-value": results["p_value"].max(),
            "Significant Windows (%)": (results["significant"].sum() / len(results))
            * 100,
            "Total Windows": len(results),
        }
        summary_data.append(summary)

    return pd.DataFrame(summary_data)
