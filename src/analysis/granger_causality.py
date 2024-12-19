# src/analysis/granger_causality.py
"""
Granger causality analysis for cryptocurrency returns.

The module provides
    Pairwise Granger Causality:
        Tests causality between all pairs of cryptocurrencies
        Finds optimal lag order for each pair
        Reports significance and p-values
    Multivariate Granger Causality:
        Uses VAR model to test causality in a multivariate setting
        Accounts for interactions between all variables
        Provides test statistics and coefficient p-values
        Automatically selects optimal lag order using AIC
    Summary Statistics:
        Counts significant causal relationships
        Shows which cryptocurrencies are most influential
        Identifies which ones are most affected by others

Key features:

    Handles missing data
    Validates inputs
    Provides detailed logging
    Flexible test statistic selection
    Configurable significance levels
    Optimal lag selection


"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import glob
from typing import Dict, Tuple, List
from src.visualization.causality_viz import CausalityVisualizer
from statsmodels.tools.eval_measures import aic, rmse
import logging

from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union



# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AutomatedGrangerAnalyzer:
    def __init__(self, data, test_type: str = "ssr_ftest"):
        """Initialize the AutomatedGrangerAnalyzer."""
        self.returns_data = data
        logger.info(f"Loaded data for {len(self.returns_data.columns)} cryptocurrencies")
        self.visualizer = CausalityVisualizer(significance_level=0.05)
        self.test_type = test_type
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input data and parameters."""
        if not isinstance(self.returns_data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        if self.returns_data.isnull().any().any():
            logger.warning("Data contains null values. These will be dropped.")
            self.returns_data = self.returns_data.dropna()

    def _test_pair_causality(
        self,
        cause: str,
        effect: str,
        max_lags: int = 100
    ) -> Tuple[Dict, int]:
        """Test Granger causality between a pair of cryptocurrencies."""
        logger.info(f"Testing causality: {cause} -> {effect}")
        
        try:
            pair_data = self.returns_data[[cause, effect]]
            logger.info(f"Testing pair with {len(pair_data)} observations")
            if len(pair_data) < max_lags + 2:
                logger.warning(f"Insufficient data for {cause}->{effect}")
                return None, None
            
            # Find optimal lag order using AIC
            model = VAR(pair_data)
            optimal_lag = model.select_order(maxlags=max_lags).aic
            logger.info(f"Optimal lag order: {optimal_lag}")
            
            # Run Granger causality test
            gc_results = grangercausalitytests(
                pair_data[[effect, cause]],
                maxlag=optimal_lag,
                verbose=False
            )

            # Extract p-values for the specified test type
            p_values = [
                gc_results[i][0][self.test_type][1] for i in range(1,optimal_lag)
            ]

            stat_values = [
                gc_results[i][0][self.test_type][0] for i in range(1,optimal_lag)
            ]

            min_p_value = min(p_values)
            optimal_lag = p_values.index(min_p_value) + 1

            return {"p_value": p_values, "optimal_lag": optimal_lag, "stat_values": stat_values, "min_p_value": min_p_value, "stat": stat_values}
            
        except Exception as e:
            logger.error(f"Error in causality test {cause}->{effect}: {str(e)}")
            return None, None

    def analyze_all_pairs(
        self,
        significance_level: float = 0.05
    ) -> pd.DataFrame:
        """Analyze Granger causality for all cryptocurrency pairs."""
        results = []
        pairs = combinations(self.returns_data.columns, 2)

        symbols = self.returns_data.columns
        total_pairs = len(symbols) * (len(symbols) - 1)
        logger.info(f"Starting analysis of {total_pairs} pairs")

        completed = 0
        for coin1, coin2 in pairs:
            completed += 1
            # Test coin1 Granger-causing coin2
            c1_to_c2 = self._test_pair_causality(coin1, coin2)

            # Test coin2 Granger-causing coin1
            c2_to_c1 = self._test_pair_causality(coin2, coin1)

            logger.info(f"Processing pair {completed}/{total_pairs}: {coin1}->{coin2}")
            results.extend(
                [
                    {
                        "cause": coin1,
                        "effect": coin2,
                        "p_value": c1_to_c2["p_value"],
                        "optimal_lag": c1_to_c2["optimal_lag"],
                        "significant": c2_to_c1["min_p_value"] < significance_level,
                        "stat": c1_to_c2["stat"]
                    },
                    {
                        "cause": coin2,
                        "effect": coin1,
                        "p_value": c2_to_c1["p_value"],
                        "optimal_lag": c2_to_c1["optimal_lag"],
                        "significant": c2_to_c1["min_p_value"] < significance_level,
                        "stat": c2_to_c1["stat"]
                    },
                ]
            )

            logger.info(f"Result: {'Significant' if results[-1]['significant'] else 'Not significant'}")
        logger.info(f"Analysis completed. Found {len(results)} valid results.")

        return pd.DataFrame(results)
    
    def run_multivariate_causality(
        self, target: str, max_lags: int = 100
    ) -> Tuple[Dict[str, float], Dict[str, List[float]], int]:
        """
        Run multivariate Granger causality analysis using VAR model.

        Args:
            target: Target cryptocurrency to test causality for
            max_lag: Maximum lag order for VAR model (default: None, uses AIC)

        Returns:
            Tuple containing:
            - Dictionary of test statistics for each variable
            - Dictionary of coefficient p-values for each variable
            - Optimal lag order
        """
        logger.info(f"Running multivariate causality analysis for target: {target}")

        # Prepare data
        data = self.returns_data.copy()

        # Determine optimal lag order if not specified
        model = VAR(data)
        opt_lag = model.select_order(maxlags=max_lags).aic
        logger.info(f"Optimal lag order determined by AIC: {opt_lag}")

        # Fit VAR model
        model = VAR(data)
        results = model.fit(maxlags=opt_lag)
        logger.info(results.params)
        logger.info(f"VAR model fitted with max lag: {opt_lag}")

        # Get test statistics and p-values for target variable
        target_idx = list(data.columns).index(target)

        # Extract coefficients and p-values for each variable
        coef_pvals = {}
        test_stats = {}

        for i, col in enumerate(data.columns):
            if col != target:
                # Get coefficients for this variable
                coefs = []
                pvals = []
                for lag in range(0, opt_lag):
                    coef_idx = i + 1 + lag * len(data.columns)
                    coefs.append(results.params.iloc[coef_idx, target_idx])
                    pvals.append(results.pvalues.iloc[coef_idx, target_idx])

                coef_pvals[col] = pvals
                # Use F-test or Chi-square test statistic
                test_stats[col] = results.test_causality(
                    target, [col], kind="f"
                ).test_statistic

                logger.info(f"Test statistics for {col} -> {target}: {test_stats[col]}")
                logger.info(f"Coefficient p-values for {col} -> {target}: {pvals}")

        return test_stats, coef_pvals, opt_lag


class GrangerCausalityAnalyzer:
    """
    A class to analyze Granger causality relationships in cryptocurrency returns.

    Attributes:
        data (pd.DataFrame): DataFrame containing the return series
        max_lags (int): Maximum number of lags to test
        test_type (str): Type of test statistic to use ('ssr_chi2test', 'ssr_ftest', etc.)
    """

    def __init__(
        self, data: pd.DataFrame, max_lags: int = 15, test_type: str = "ssr_ftest"
    ):
        """
        Initialize the GrangerCausalityAnalyzer.

        Args:
            data: DataFrame with cryptocurrency returns (each column is a crypto)
            max_lags: Maximum number of lags to test for causality
            test_type: Type of test statistic ('ssr_chi2test', 'ssr_ftest',
                      'ssr_chi2test', 'lrtest')
        """
        self.data = data
        self.max_lags = max_lags
        self.test_type = test_type
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input data and parameters."""
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        if self.data.isnull().any().any():
            logger.warning("Data contains null values. These will be dropped.")
            self.data = self.data.dropna()

        valid_tests = ["ssr_chi2test", "ssr_ftest", "lrtest", "params_ftest"]
        if self.test_type not in valid_tests:
            raise ValueError(f"test_type must be one of {valid_tests}")

    def run_pairwise_causality(self, significance_level: float = 0.05) -> pd.DataFrame:
        """
        Run pairwise Granger causality tests between all pairs of cryptocurrencies.

        Args:
            significance_level: P-value threshold for significance

        Returns:
            DataFrame containing test results for each pair
        """
        results = []
        pairs = combinations(self.data.columns, 2)

        for coin1, coin2 in pairs:
            # Test coin1 Granger-causing coin2
            c1_to_c2 = self._test_pair_causality(coin1, coin2)

            # Test coin2 Granger-causing coin1
            c2_to_c1 = self._test_pair_causality(coin2, coin1)

            results.extend(
                [
                    {
                        "cause": coin1,
                        "effect": coin2,
                        "p_value": c1_to_c2["p_value"],
                        "optimal_lag": c1_to_c2["optimal_lag"],
                        "significant": c2_to_c1["min_p_value"] < significance_level,
                        "stat": c1_to_c2["stat"]
                    },
                    {
                        "cause": coin2,
                        "effect": coin1,
                        "p_value": c2_to_c1["p_value"],
                        "optimal_lag": c2_to_c1["optimal_lag"],
                        "significant": c2_to_c1["min_p_value"] < significance_level,
                        "stat": c2_to_c1["stat"],
                    },
                ]
            )
        
            logger.info(f"Result: {'Significant' if results[-1]['significant'] else 'Not significant'}")
        logger.info(f"Analysis completed. Found {len(results)} valid results.")

        return pd.DataFrame(results)

    def _test_pair_causality(
        self, cause: str, effect: str
    ) -> Dict[str, Union[float, int]]:
        """
        Test Granger causality between a pair of cryptocurrencies.

        Args:
            cause: Name of potential causing variable
            effect: Name of potential effect variable

        Returns:
            Dictionary containing test results
        """
        data = self.data[[cause, effect]].dropna()
        test_results = grangercausalitytests(data, maxlag=self.max_lags, verbose=False)

        # Extract p-values for the specified test type
        p_values = [
            test_results[i + 1][0][self.test_type][1] for i in range(self.max_lags)
        ]

        stat_values = [
            test_results[i + 1][0][self.test_type][0] for i in range(self.max_lags)
        ]

        min_p_value = min(p_values)
        optimal_lag = p_values.index(min_p_value) + 1

        return {"p_value": p_values, "optimal_lag": optimal_lag, "stat_values": stat_values, "min_p_value": min_p_value, "stat": stat_values}

    def run_multivariate_causality(
        self, target: str, max_lag: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, List[float]], int]:
        """
        Run multivariate Granger causality analysis using VAR model.

        Args:
            target: Target cryptocurrency to test causality for
            max_lag: Maximum lag order for VAR model (default: None, uses AIC)

        Returns:
            Tuple containing:
            - Dictionary of test statistics for each variable
            - Dictionary of coefficient p-values for each variable
            - Optimal lag order
        """
        # Prepare data
        data = self.data.copy()

        # Determine optimal lag order if not specified
        if max_lag is None:
            model = VAR(data)
            max_lag = model.select_order(maxlags=self.max_lags).aic

        # Fit VAR model
        model = VAR(data)
        results = model.fit(maxlags=max_lag)

        # Get test statistics and p-values for target variable
        target_idx = list(data.columns).index(target)

        # Extract coefficients and p-values for each variable
        coef_pvals = {}
        test_stats = {}

        for i, col in enumerate(data.columns):
            if col != target:
                # Get coefficients for this variable
                coefs = []
                pvals = []
                for lag in range(max_lag):
                    coef_idx = i + lag * len(data.columns)
                    coefs.append(results.params[coef_idx][target_idx])
                    pvals.append(results.pvalues[coef_idx][target_idx])

                coef_pvals[col] = pvals
                # Use F-test or Chi-square test statistic
                test_stats[col] = results.test_causality(
                    target, [col], kind="f"
                ).test_statistic

        return test_stats, coef_pvals, max_lag

    def get_summary_statistics(self, significance_level: float = 0.05) -> pd.DataFrame:
        """
        Generate summary statistics of causality relationships.

        Args:
            significance_level: P-value threshold for significance

        Returns:
            DataFrame with summary statistics
        """
        results = self.run_pairwise_causality(significance_level)

        summary = pd.DataFrame(index=self.data.columns)
        summary["causes_count"] = (
            results[results["significant"]].groupby("cause").size()
        )
        summary["affected_by_count"] = (
            results[results["significant"]].groupby("effect").size()
        )
        summary["total_relationships"] = (
            summary["causes_count"] + summary["affected_by_count"]
        )

        return (
            summary.fillna(0)
            .astype(int)
            .sort_values("total_relationships", ascending=False)
        )
