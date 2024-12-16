# examples/automated_gc.py
"""
Key features of this implementation:

    Automated Lag Selection:

    Uses VAR model to determine optimal lag structure
    Applies AIC criterion to select best lag order
    Considers both variables simultaneously in the model

    Comprehensive Testing:

    Returns both Chi-square and F-test statistics
    Provides p-values for hypothesis testing
    Includes interpretation of results

    Advantages of this Approach:

    More robust than selecting lags for x and y separately
    Accounts for potential interactions between variables
    Reduces risk of overfitting or underfitting

    Limitations to Consider:

    Assumes stationarity of the time series
    May need larger sample sizes for higher lag orders
    Sensitive to the maximum lag parameter

The code allows for fully automated testing while still giving you
control over key parameters like maximum lags and significance levels.
 This makes it suitable for both individual tests and large-scale analyses
 of multiple variable pairs.

---


"""

import os
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.visualization.causality_viz import CausalityVisualizer
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomatedGrangerAnalyzer:
    def __init__(self, data_dir: str = "../data/processed/"):
        """
        Initialize analyzer with data directory path.

        Args:
            data_dir: Directory containing processed parquet files
        """
        self.data_dir = data_dir
        self.returns_data = self._load_data()
        self.visualizer = CausalityVisualizer(significance_level=0.05)

    def _load_data(self) -> pd.DataFrame:
        """Load and prepare return data from parquet files."""
        all_returns = {}

        for file in glob.glob(os.path.join(self.data_dir, "*.parquet")):
            symbol = os.path.basename(file).split("_")[0]
            df = pq.read_table(file).to_pandas()
            # Calculate log returns if not already present
            if "log_returns" not in df.columns:
                df["log_returns"] = np.log(df["close"]).diff()
            all_returns[symbol] = df["log_returns"]

        returns_df = pd.DataFrame(all_returns).dropna()
        return returns_df

    def select_optimal_lag(self, data: pd.DataFrame, max_lags: int = 10) -> int:
        """
        Select optimal lag order using AIC criterion.

        Args:
            data: DataFrame containing the variables
            max_lags: Maximum number of lags to consider

        Returns:
            int: Optimal lag order
        """
        model = VAR(data)
        results = model.select_order(maxlags=max_lags)
        return results.aic.argmin() + 1  # Add 1 since lags are 1-based

    def test_pair_causality(
        self, cause: str, effect: str, max_lags: int = 10
    ) -> Tuple[Dict, int]:
        """
        Test Granger causality between a pair of cryptocurrencies.

        Args:
            cause: Name of potential causing variable
            effect: Name of potential effect variable
            max_lags: Maximum number of lags to test

        Returns:
            Tuple containing test results and optimal lag order
        """
        # Prepare data
        pair_data = self.returns_data[[cause, effect]].dropna()

        # Find optimal lag order
        optimal_lag = self.select_optimal_lag(pair_data, max_lags)

        # Run Granger causality test
        gc_results = grangercausalitytests(
            pair_data[[effect, cause]],  # effect must come first
            maxlag=optimal_lag,
            verbose=False,
        )

        # Extract results
        test_stats = {}
        for test_type in ["ssr_chi2test", "ssr_ftest"]:
            test_stats[test_type] = {
                "stat": gc_results[optimal_lag][0][test_type][0],
                "pvalue": gc_results[optimal_lag][0][test_type][1],
            }

        return test_stats, optimal_lag

    def analyze_all_pairs(self, significance_level: float = 0.05) -> pd.DataFrame:
        """
        Analyze Granger causality for all cryptocurrency pairs.

        Returns:
            DataFrame with causality test results
        """
        results = []
        symbols = self.returns_data.columns

        for cause in symbols:
            for effect in symbols:
                if cause != effect:
                    test_stats, opt_lag = self.test_pair_causality(cause, effect)

                    results.append(
                        {
                            "cause": cause,
                            "effect": effect,
                            "optimal_lag": opt_lag,
                            "f_stat": test_stats["ssr_ftest"]["stat"],
                            "p_value": test_stats["ssr_ftest"]["pvalue"],
                            "significant": test_stats["ssr_ftest"]["pvalue"]
                            < significance_level,
                        }
                    )

        return pd.DataFrame(results)

    def visualize_results(self, results: pd.DataFrame) -> None:
        """Create visualizations of causality results."""
        # Create heatmap of p-values
        self.visualizer.plot_causality_heatmap(
            results, title="Automated Granger Causality P-values"
        )

        # Create network diagram
        self.visualizer.plot_causality_network(
            results, title="Automated Granger Causality Network"
        )


def main():
    """Main analysis pipeline."""
    try:
        # Initialize analyzer
        analyzer = AutomatedGrangerAnalyzer()

        # Run analysis
        logger.info("Running automated Granger causality analysis...")
        results = analyzer.analyze_all_pairs()

        if results is None or len(results) == 0:
            print("\nNo Granger causality results were generated.")
            return

        # Print all results first for debugging
        print("\nAll Granger Causality Tests:")
        print(results)

        # Safely filter significant results
        if "significant" in results.columns:
            significant_mask = results["significant"] == True
            significant_results = results[significant_mask]

            if len(significant_results) > 0:
                print("\nSignificant Granger Causality Relationships:")
                for _, row in significant_results.iterrows():
                    print(f"{row['cause']} -> {row['effect']}")
                    print(f"  Optimal lag: {row['optimal_lag']}")
                    print(f"  F-statistic: {row['f_stat']:.4f}")
                    print(f"  P-value: {row['p_value']:.4f}\n")
            else:
                print("\nNo significant Granger causality relationships found.")
        else:
            print("\nError: 'significant' column not found in results.")
            print("Available columns:", results.columns.tolist())

    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
